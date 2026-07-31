from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
import yaml

from .artifacts import ArtifactRun, git_fingerprint
from .config import ExperimentConfig, ModelConfig
from .general_corpus import write_synthetic_token_memmap
from .hashing import hash_mapping, optimizer_state_signature, parameter_signature_hash
from .interventions import GateIntervention
from .locality import (
    active_key_covariance,
    covariance_aware_direction,
    protected_nullspace_direction,
    worst_non_target_response,
)
from .model import (
    CatenaLM,
    assert_matched_models,
    build_paired_models,
    cross_entropy_loss,
)
from .oracle_control import (
    OracleLevel,
    first_substantial_rescue,
    fit_bounded_erase_write,
    operation_gate_target,
)
from .recurrent_mixer import (
    TransactionalDeltaMixer,
    optimized_backend_diagnostics,
    optimized_backend_metadata,
    reset_optimized_backend_diagnostics,
)
from .statistics import bootstrap_interval, did_by_seed, exact_sign_flip_pvalue
from .systems_boundary import PolicyMeasurement, break_even_queries, quality_constrained_pareto
from .tokenizer import ByteTokenizer
from .trainer import (
    compare_optimizer_signatures,
    cycle_tensor_batches,
    finite_training_metrics,
    make_optimizer,
    measure_checkpoint_io,
    train_non_evidence_smoke,
    train_reference_steps,
)
from .transactional_stream import (
    Operation,
    audit_split_disjointness,
    generate_grid,
)


def _parser(experiment: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"CATENA v8.1 {experiment} reference/contract entry point"
    )
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dependency-report", action="append", default=[])
    parser.add_argument("--backend-manifest")
    parser.add_argument("--protocol-lock")
    parser.add_argument("--tokenizer-manifest")
    parser.add_argument("--corpus-manifest")
    parser.add_argument("--non-evidence-smoke", action="store_true")
    parser.add_argument("--candidate-id", default="d512_ctx4096")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=100)
    return parser


def _load_raw_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    contract = ExperimentConfig.load(path)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return contract, raw


def _guard_mode(args: argparse.Namespace, experiment: str) -> None:
    if args.non_evidence_smoke and not args.dry_run:
        raise SystemExit(
            f"{experiment}: --non-evidence-smoke requires --dry-run and a fresh /tmp root"
        )
    if args.dry_run:
        return
    if not args.allow_main:
        raise SystemExit(
            f"{experiment}: non-dry execution requires --allow-main and upstream scientific locks"
        )
    required = {
        "--backend-manifest": args.backend_manifest,
        "--protocol-lock": args.protocol_lock,
        "--tokenizer-manifest": args.tokenizer_manifest,
        "--corpus-manifest": args.corpus_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"{experiment}: scientific execution is missing {', '.join(missing)}")
    manifest = json.loads(Path(args.backend_manifest).read_text(encoding="utf-8"))
    if experiment == "e26a_operator_data_gate":
        if not manifest.get("e26a_candidate_capable", False):
            raise SystemExit(f"{experiment}: backend_manifest.e26a_candidate_capable is not true")
    elif not manifest.get("scientific_main_capable", False):
        raise SystemExit(f"{experiment}: backend_manifest.scientific_main_capable is not true")
    if experiment != "e26a_operator_data_gate":
        raise SystemExit(
            f"{experiment}: scientific MAIN remains blocked until the preceding E26 "
            "dependency report has been frozen"
        )


def _base_files(
    run: ArtifactRun,
    *,
    config_path: str,
    raw_config: dict[str, Any],
    model: CatenaLM | None = None,
    tokenizer_manifest: dict[str, Any] | None = None,
    extra_data_manifest: dict[str, Any] | None = None,
    backend_manifest: dict[str, Any] | None = None,
    model_manifest_overrides: dict[str, Any] | None = None,
) -> None:
    config_bytes = Path(config_path).read_bytes()
    protocol_lock = {
        "schema_version": "catena-v8.1",
        "experiment": raw_config["experiment"],
        "stage": raw_config["stage"],
        "locked": False,
        "lock_utc": None,
        "source_hash": "0" * 64,
        "config_hash": hash_mapping(raw_config),
        "primary_question": "NON_EVIDENCE_REFERENCE_VALIDATION",
        "primary_estimand": "NOT_APPLICABLE_TO_SCIENTIFIC_CLAIM",
        "inference_unit": "diagnostic_episode",
        "registered_dispositions": raw_config.get("registered_dispositions", []),
        "config_bytes_hash": hash_mapping(list(config_bytes)),
    }
    run.write("protocol_lock.json", protocol_lock)
    data_manifest = {
        "schema_version": "catena-v8.1",
        "generator_version": "v8.1-reference",
        "tokenizer": tokenizer_manifest or {"id": "none", "hash": "0" * 64, "vocab_size": 0},
        "general_corpus": {
            "revision": "SYNTHETIC_NON_EVIDENCE",
            "token_file_hash": "0" * 64,
        },
        "transaction_splits": {},
        "split_audit": {"disjoint": True, "duplicates": 0, "answer_leakage": 0},
        "manifest_hash": "0" * 64,
    }
    if extra_data_manifest:
        data_manifest.update(extra_data_manifest)
    data_manifest["manifest_hash"] = hash_mapping(
        {key: value for key, value in data_manifest.items() if key != "manifest_hash"}
    )
    run.write("data_manifest.json", data_manifest)
    if model is not None:
        runtime = model.initial_runtime_state(
            1, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
        )
        model_manifest = {
            "schema_version": "catena-v8.1",
            "variant": model.config.variant,
            "model_config": model.config.to_dict(),
            "parameter_count": model.parameter_count(),
            "parameter_signature_hash": parameter_signature_hash(model),
            "initialization_digest": model.initialization_digest(),
            "pair_id": "reference-smoke-pair",
            "optimizer_state_signature": None,
            "recurrent_state_shape": list(runtime.recurrent[0].matrix.shape),
            "recurrent_state_bytes": sum(
                state.matrix.numel() * state.matrix.element_size() for state in runtime.recurrent
            ),
            "checkpoint": None,
        }
        if model_manifest_overrides:
            model_manifest.update(model_manifest_overrides)
        run.write("model_manifest.json", model_manifest)
    else:
        run.write(
            "model_manifest.json",
            {
                "schema_version": "catena-v8.1",
                "not_applicable": True,
                "reason": "This dry-run stage does not instantiate a language model.",
            },
        )
    run.write(
        "backend_manifest.json",
        backend_manifest
        or {
            "schema_version": "catena-v8.1",
            "backend_id": "reference_python",
            "backend_type": "REFERENCE_PYTHON",
            "source_commit": "PACKET_REFERENCE",
            "kernel_commit": None,
            "full_chunk_relative_l2": 0.0,
            "state_carry_relative_l2": 0.0,
            "bf16_fp32_relative_l2": None,
            "gradient_finite": True,
            "state_clone_no_alias": True,
            "graph_break_count": 0,
            "fallback_count": 0,
            "scientific_main_capable": False,
        },
    )
    for name in ("training_metrics.jsonl", "evaluation_metrics.jsonl", "seed_effects.jsonl"):
        (run.run_dir / name).touch()


def _gate(name: str, passed: bool, observed: Any, criterion: Any, note: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "criterion": criterion,
        "note": note,
    }


def _summary(experiment: str, disposition: str, gates: list[dict[str, Any]]) -> str:
    lines = [
        f"# {experiment} reference validation",
        "",
        "- Evidence tier: `NON_EVIDENCE_VALIDATION`",
        "- Scientific evidence: `false`",
        f"- Disposition: `{disposition}`",
        "",
        "## Gates",
        "",
        "| Gate | Pass | Observed |",
        "|---|---:|---|",
    ]
    for gate in gates:
        observed = json.dumps(gate["observed"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{gate['name']}` | {gate['passed']} | `{observed}` |")
    lines += [
        "",
        (
            "이 결과는 packet의 equation·schema·artifact smoke만 검증하며 "
            "scientific claim을 열지 않는다."
        ),
    ]
    return "\n".join(lines)


def _final_report(
    run: ArtifactRun,
    *,
    experiment: str,
    disposition: str,
    gates: list[dict[str, Any]],
    status: str = "PASS",
    allowed_claim: str = "Reference implementation and contracts execute in non-evidence mode.",
) -> dict[str, Any]:
    return {
        "schema_version": "catena-v8.1",
        "experiment": experiment,
        "run_id": run.run_id,
        "run_mode": "DRY_RUN",
        "status": status,
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_VALIDATION",
        "disposition": disposition,
        "allowed_claim": allowed_claim,
        "forbidden_claims": [
            "autoregressive LM transfer",
            "official GDN2/KDA correspondence",
            "agent or production superiority",
        ],
        "gates": gates,
        "artifacts": {"run_dir": str(run.run_dir)},
        "upstream_dependencies": [],
    }


def _tiny_sequences(tokenizer: ByteTokenizer, length: int = 32) -> list[list[int]]:
    texts = [
        "alpha record changed after a verified update",
        "keep unrelated beta while revising gamma",
        "the current action must not use a stale version",
        "write a replacement and preserve the audit record",
    ]
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        ids = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
        sequences.append(ids)
    return sequences


def _force_equal_raw_gate_halves(model: CatenaLM) -> None:
    with torch.no_grad():
        for block in model.blocks:
            if not block.is_recurrent:
                continue
            mixer = block.mixer
            if not isinstance(mixer, TransactionalDeltaMixer):
                raise TypeError("Recurrent block does not contain a delta mixer")
            head = mixer.gate_head
            half = head.weight.shape[0] // 2
            head.weight[half:].copy_(head.weight[:half])
            head.bias[half:].copy_(head.bias[:half])


def _candidate_model_config(
    raw: dict[str, Any],
    *,
    candidate_id: str,
    variant: str = "dual_delta_lm",
) -> ModelConfig:
    candidates = raw.get("model_candidates")
    if not isinstance(candidates, list):
        raise ValueError("E26a config lacks model_candidates")
    selected = next(
        (
            dict(candidate)
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("id") == candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"Unknown E26a model candidate: {candidate_id!r}")
    selected.pop("id", None)
    selected.update(
        {
            "variant": variant,
            "backend_id": "compiled_scan",
            # Dispatch is selected by backend_id. Capability remains false
            # until the complete E26a gate produces a frozen backend manifest.
            "backend_scientific_main_capable": False,
            "optimized_chunk_size": 32,
        }
    )
    return ModelConfig.from_mapping(selected)


def _random_batches(
    *,
    vocab_size: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    seed: int,
) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    while True:
        yield torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            generator=generator,
            device=device,
        )


def _paired_compiled_step(
    tied: CatenaLM,
    dual: CatenaLM,
    batch: torch.Tensor,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizers = (make_optimizer(tied), make_optimizer(dual))
    for model, optimizer in zip((tied, dual), optimizers, strict=True):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss = cross_entropy_loss(output.logits, batch)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return optimizers


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).norm()
    denominator = right.float().norm().clamp_min(torch.finfo(torch.float32).eps)
    return float((numerator / denominator).item())


def _hybrid_runtime_state_errors(
    observed: Any,
    expected: Any,
) -> tuple[float, float, bool]:
    recurrent_error = max(
        (
            _relative_l2(left.matrix, right.matrix)
            for left, right in zip(observed.recurrent, expected.recurrent, strict=True)
        ),
        default=0.0,
    )
    attention_error = max(
        (
            max(
                _relative_l2(left.key, right.key),
                _relative_l2(left.value, right.value),
            )
            for left, right in zip(observed.attention, expected.attention, strict=True)
        ),
        default=0.0,
    )
    attention_metadata_equal = all(
        left.length == right.length
        and left.write_index == right.write_index
        and torch.equal(left.positions, right.positions)
        for left, right in zip(observed.attention, expected.attention, strict=True)
    )
    return recurrent_error, attention_error, attention_metadata_equal


def _run_e26a_non_evidence_smoke(
    args: argparse.Namespace,
    raw: dict[str, Any],
    config_path: str,
) -> int:
    """Run the explicitly authorized 100-step feasibility smoke on one GPU."""

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("E26a 100-step non-evidence smoke requires --device cuda")
    if args.measured_steps != 100:
        raise SystemExit("The registered non-evidence smoke requires exactly 100 measured steps")
    locked_warmup = int(raw["throughput"]["warmup_steps"])
    if args.warmup_steps != locked_warmup:
        raise SystemExit(f"The E26a smoke requires the locked {locked_warmup} warmup steps")
    if args.batch_size <= 0 or args.sequence_length <= 1:
        raise SystemExit("Smoke batch size and sequence length must be positive")
    config = _candidate_model_config(raw, candidate_id=args.candidate_id)
    if args.sequence_length > config.context_length:
        raise SystemExit("Smoke sequence length exceeds the selected candidate context")

    run = ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    reset_optimized_backend_diagnostics()
    tied, dual = build_paired_models(config, seed=26_000, device=device)
    assert_matched_models(tied, dual)
    parameter_count = dual.parameter_count()
    init_digest = dual.initialization_digest()
    signature_hash = parameter_signature_hash(dual)
    paired_initialization = tied.initialization_digest() == init_digest
    paired_signature = parameter_signature_hash(tied) == signature_hash
    batches = _random_batches(
        vocab_size=config.vocab_size,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
        seed=26_001,
    )
    paired_batch = next(batches)
    tied_optimizer, dual_optimizer = _paired_compiled_step(
        tied,
        dual,
        paired_batch,
    )
    optimizer_match = compare_optimizer_signatures(tied_optimizer, dual_optimizer)
    del tied_optimizer
    del tied
    torch.cuda.empty_cache()

    summary, dual_optimizer = train_non_evidence_smoke(
        dual,
        batches,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        optimizer=dual_optimizer,
    )
    checkpoint_io = measure_checkpoint_io(
        dual,
        dual_optimizer,
        run.checkpoint_dir() / "non_evidence_smoke.pt",
    )

    # Hybrid full-vs-state-carry parity is measured on the trained candidate.
    dual.eval()
    parity_length = min(63, args.sequence_length)
    parity_ids = next(
        _random_batches(
            vocab_size=config.vocab_size,
            batch_size=1,
            sequence_length=parity_length,
            device=device,
            seed=26_002,
        )
    )
    split = parity_ids.shape[1] // 2
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full = dual(parity_ids)
        prefix = dual(parity_ids[:, :split])
        suffix = dual(parity_ids[:, split:], prefix.runtime_state)
    hybrid_logit_error = _relative_l2(
        torch.cat((prefix.logits, suffix.logits), dim=1),
        full.logits,
    )
    (
        hybrid_recurrent_error,
        hybrid_attention_error,
        hybrid_attention_metadata_equal,
    ) = _hybrid_runtime_state_errors(
        suffix.runtime_state,
        full.runtime_state,
    )
    clone = full.runtime_state.clone(detach=True)
    clone_no_alias = not bool(set(full.runtime_state.storage_ptrs()) & set(clone.storage_ptrs()))

    # Check the actual d512 recurrent operator rather than a tiny-only proxy.
    candidate_mixer = dual.blocks[config.recurrent_layers[0]].mixer
    if not isinstance(candidate_mixer, TransactionalDeltaMixer):
        raise TypeError("Selected recurrent layer does not contain a delta mixer")
    probe_hidden_fp32 = torch.randn(
        1,
        17,
        config.d_model,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        fp32_reference, fp32_reference_state, _ = candidate_mixer.forward_reference(
            probe_hidden_fp32
        )
        fp32_optimized, fp32_optimized_state, baseline_trace = candidate_mixer.forward_optimized(
            probe_hidden_fp32,
            chunk_size=config.optimized_chunk_size,
            compiler="inductor",
            return_gate_trace=True,
        )
    fp32_error = _relative_l2(fp32_optimized, fp32_reference)
    fp32_max_abs = float((fp32_optimized.float() - fp32_reference.float()).abs().max().item())
    fp32_state_error = _relative_l2(
        fp32_optimized_state.matrix,
        fp32_reference_state.matrix,
    )
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        bf16_optimized, bf16_state, _ = candidate_mixer.forward_optimized(
            probe_hidden_fp32,
            chunk_size=config.optimized_chunk_size,
            compiler="inductor",
        )
    bf16_fp32_error = _relative_l2(bf16_optimized, fp32_optimized)
    bf16_state_fp32_error = _relative_l2(
        bf16_state.matrix,
        fp32_optimized_state.matrix,
    )

    intervention_mask = torch.zeros(17, dtype=torch.bool, device=device)
    intervention_mask[5:9] = True
    with torch.no_grad():
        _, _, intervention_trace = candidate_mixer.forward_optimized(
            probe_hidden_fp32,
            chunk_size=config.optimized_chunk_size,
            compiler="inductor",
            gate_intervention=GateIntervention(
                erase_scale=0.0,
                token_mask=intervention_mask,
            ),
            return_gate_trace=True,
        )
    if baseline_trace is None or intervention_trace is None:
        raise RuntimeError("Optimized gate trace was not returned")
    intervention_outside = ~intervention_mask
    intervention_confined = bool(
        torch.equal(
            baseline_trace.erase[:, intervention_outside],
            intervention_trace.erase[:, intervention_outside],
        )
        and torch.equal(
            baseline_trace.write[:, intervention_outside],
            intervention_trace.write[:, intervention_outside],
        )
    )
    intervention_active = bool(
        torch.any(
            baseline_trace.erase[:, intervention_mask]
            != intervention_trace.erase[:, intervention_mask]
        ).item()
    )
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in dual.parameters()
    )
    diagnostics = optimized_backend_diagnostics()
    static_backend = optimized_backend_metadata(
        device=device,
        compiler="inductor",
        chunk_size=config.optimized_chunk_size,
    )
    numerical_thresholds = raw["backend_gates"]
    fp32_relative_limit = float(numerical_thresholds["fp32_full_chunk_relative_l2_max"])
    fp32_absolute_limit = float(numerical_thresholds["fp32_full_chunk_max_abs_max"])
    bf16_limit = float(numerical_thresholds["bf16_fp32_relative_l2_max"])
    grad_min = float(numerical_thresholds["gradient_norm_min"])
    grad_max = float(numerical_thresholds["gradient_norm_max"])
    gates = [
        _gate(
            "actual_parameter_range",
            35_000_000 <= parameter_count <= 50_000_000,
            parameter_count,
            "[35M,50M]",
        ),
        _gate("paired_parameter_signature", paired_signature, signature_hash, "identical"),
        _gate("paired_initialization_digest", paired_initialization, init_digest, "identical"),
        _gate(
            "optimizer_state_shape_match", optimizer_match.matched, optimizer_match.matched, True
        ),
        _gate(
            "hybrid_full_chunk_relative_l2",
            hybrid_logit_error <= bf16_limit,
            hybrid_logit_error,
            bf16_limit,
        ),
        _gate(
            "hybrid_recurrent_state_carry_relative_l2",
            hybrid_recurrent_error <= bf16_limit,
            hybrid_recurrent_error,
            bf16_limit,
        ),
        _gate(
            "hybrid_attention_state_carry_relative_l2",
            hybrid_attention_error <= bf16_limit,
            hybrid_attention_error,
            bf16_limit,
        ),
        _gate(
            "hybrid_attention_metadata_equal",
            hybrid_attention_metadata_equal,
            hybrid_attention_metadata_equal,
            True,
        ),
        _gate("state_clone_no_alias", clone_no_alias, clone_no_alias, True),
        _gate(
            "fp32_reference_optimized_relative_l2",
            fp32_error <= fp32_relative_limit,
            fp32_error,
            fp32_relative_limit,
        ),
        _gate(
            "fp32_reference_optimized_max_abs",
            fp32_max_abs <= fp32_absolute_limit,
            fp32_max_abs,
            fp32_absolute_limit,
        ),
        _gate(
            "fp32_reference_optimized_state_l2",
            fp32_state_error <= fp32_relative_limit,
            fp32_state_error,
            fp32_relative_limit,
        ),
        _gate(
            "bf16_fp32_relative_l2",
            bf16_fp32_error <= bf16_limit,
            bf16_fp32_error,
            bf16_limit,
        ),
        _gate(
            "bf16_fp32_state_relative_l2",
            bf16_state_fp32_error <= bf16_limit,
            bf16_state_fp32_error,
            bf16_limit,
        ),
        _gate("gradient_finite", gradient_finite, gradient_finite, True),
        _gate(
            "gradient_norm_range",
            grad_min <= summary.max_grad_norm <= grad_max,
            summary.max_grad_norm,
            [grad_min, grad_max],
        ),
        _gate(
            "intervention_hook_confined",
            intervention_confined and intervention_active,
            {
                "outside_equal": intervention_confined,
                "selected_changed": intervention_active,
            },
            {"outside_equal": True, "selected_changed": True},
        ),
        _gate(
            "finite_100_step_training",
            all(math.isfinite(float(value)) for value in summary.to_dict().values()),
            summary.to_dict(),
            "all finite",
        ),
        _gate(
            "optimized_graph_breaks",
            diagnostics["graph_break_count"] == 0,
            diagnostics["graph_break_count"],
            0,
        ),
        _gate(
            "optimized_fallbacks",
            diagnostics["fallback_count"] == 0,
            diagnostics["fallback_count"],
            0,
        ),
    ]
    numerical_pass = all(gate["passed"] for gate in gates)
    backend_manifest = {
        "schema_version": "catena-v8.1",
        **static_backend,
        "backend_type": "TORCH_COMPILED",
        "source_commit": git_fingerprint(Path.cwd())["head"],
        "source_fingerprint": run.source_fingerprint,
        "kernel_commit": None,
        "full_chunk_relative_l2": fp32_error,
        "state_carry_relative_l2": max(
            hybrid_recurrent_error,
            hybrid_attention_error,
        ),
        "bf16_fp32_relative_l2": bf16_fp32_error,
        "gradient_finite": gradient_finite,
        "state_clone_no_alias": clone_no_alias,
        "e26a_candidate_capable": numerical_pass,
        "e26a_gate_capable": False,
        "parity_verified": False,
        # Only the complete E26a registered grid may promote these fields.
        "scientific_main_capable": False,
        "observed_diagnostics": diagnostics,
        "observed_smoke": {
            "fp32_output_relative_l2": fp32_error,
            "fp32_state_relative_l2": fp32_state_error,
            "bf16_fp32_output_relative_l2": bf16_fp32_error,
            "bf16_fp32_state_relative_l2": bf16_state_fp32_error,
            "hybrid_output_state_carry_relative_l2": hybrid_logit_error,
            "hybrid_recurrent_state_carry_relative_l2": hybrid_recurrent_error,
            "hybrid_attention_state_carry_relative_l2": hybrid_attention_error,
            "hybrid_attention_metadata_equal": hybrid_attention_metadata_equal,
        },
        "claim_boundary": "NON_EVIDENCE_100_STEP_SMOKE_ONLY",
    }
    corpus = write_synthetic_token_memmap(run.run_dir / "synthetic_corpus")
    tokenizer = ByteTokenizer()
    tokenizer_record = tokenizer.manifest()
    _base_files(
        run,
        config_path=config_path,
        raw_config=raw,
        model=dual,
        tokenizer_manifest={
            "id": tokenizer_record.tokenizer_id,
            "hash": tokenizer_record.manifest_hash,
            "vocab_size": tokenizer_record.vocab_size,
        },
        extra_data_manifest={
            "general_corpus": {
                "revision": corpus.corpus_revision,
                "token_file_hash": corpus.token_file_sha256,
            },
            "transaction_splits": {},
            "split_audit": {
                "disjoint": True,
                "duplicates": 0,
                "answer_leakage": 0,
                "not_applicable_reason": "random-token throughput smoke",
            },
        },
        backend_manifest=backend_manifest,
        model_manifest_overrides={
            "initialization_digest": init_digest,
            "pair_id": "e26a-non-evidence-paired-smoke",
            "optimizer_state_signature": optimizer_state_signature(dual_optimizer),
            "checkpoint": {
                "path": checkpoint_io["path"],
                "sha256": checkpoint_io["sha256"],
                "bytes": checkpoint_io["bytes"],
                "scientific_evidence": False,
            },
        },
    )
    run.append(
        "training_metrics.jsonl",
        [
            {
                "experiment": "e26a_operator_data_gate",
                "run_mode": "DRY_RUN",
                "evidence_tier": "NON_EVIDENCE_VALIDATION",
                **summary.to_dict(),
            }
        ],
    )
    disposition = (
        "NON_EVIDENCE_100_STEP_SMOKE_PASS"
        if numerical_pass
        else "NON_EVIDENCE_100_STEP_SMOKE_NUMERICAL_FAIL"
    )
    report = _final_report(
        run,
        experiment="e26a_operator_data_gate",
        disposition=disposition,
        gates=gates,
        status="PASS" if numerical_pass else "FAIL",
        allowed_claim=(
            "The integrated candidate completed a fixed 100-step non-evidence "
            "GPU feasibility smoke; no LM or E26 scientific claim is opened."
        ),
    )
    report.update(
        {
            "candidate_id": args.candidate_id,
            "parameter_count": parameter_count,
            "parameter_signature_hash": signature_hash,
            "initialization_digest": init_digest,
            "throughput": summary.to_dict(),
            "optimizer_updates_total": 1 + args.warmup_steps + args.measured_steps,
            "token_exposure_total": (
                (1 + args.warmup_steps + args.measured_steps)
                * args.batch_size
                * args.sequence_length
            ),
            "checkpoint_io": checkpoint_io,
            "backend_manifest_candidate": backend_manifest,
            "scientific_main_started": False,
            "scientific_data_ready": False,
        }
    )
    run.finalize(report, _summary("E26a 100-step smoke", disposition, gates))
    print(run.run_dir)
    return 0 if numerical_pass else 1


def _run_e26a(args: argparse.Namespace, raw: dict[str, Any], config_path: str) -> int:
    if args.non_evidence_smoke:
        return _run_e26a_non_evidence_smoke(args, raw, config_path)
    if not args.dry_run:
        raise SystemExit(
            "E26a scientific execution is not implemented in the reference driver. "
            "The optimized candidate must first pass the complete registered E26a "
            "operator/data/numerical/throughput gate without a scientific training run."
        )
    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    base = ModelConfig.tiny_reference()
    tied, dual = build_paired_models(base, seed=26000, device=device)
    assert_matched_models(tied, dual)
    run = ArtifactRun(
        experiment="e26a_operator_data_gate",
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    episodes = list(
        generate_grid(
            seed=7,
            splits=["train", "validation", "main_test", "heldout_domain"],
            domains=["access_control"],
            operations=list(Operation),
            items_per_cell=2,
            distractor_units=1,
        )
    )
    split_audit = audit_split_disjointness(episodes)
    corpus = write_synthetic_token_memmap(run.run_dir / "synthetic_corpus")
    _base_files(
        run,
        config_path=config_path,
        raw_config=raw,
        model=dual,
        tokenizer_manifest={
            "id": tokenizer.manifest().tokenizer_id,
            "hash": tokenizer.manifest().manifest_hash,
            "vocab_size": tokenizer.vocab_size,
        },
        extra_data_manifest={
            "general_corpus": {
                "revision": corpus.corpus_revision,
                "token_file_hash": corpus.token_file_sha256,
            },
            "transaction_splits": split_audit["split_counts"],
            "split_audit": {
                "disjoint": split_audit["disjoint"],
                "duplicates": len(split_audit["duplicates"]),
                "answer_leakage": sum(len(v) for v in split_audit["validation_errors"].values()),
            },
        },
    )
    input_ids = torch.randint(0, base.vocab_size, (2, 25), device=device)
    with torch.no_grad():
        full = dual(input_ids, chunked_reference=False)
        chunk = dual(input_ids, chunked_reference=True)
    full_chunk_error = float((full.logits - chunk.logits).float().norm().item())
    state_error = max(
        float((left.matrix - right.matrix).float().norm().item())
        for left, right in zip(
            full.runtime_state.recurrent, chunk.runtime_state.recurrent, strict=True
        )
    )
    clone = full.runtime_state.clone()
    clone_no_alias = not bool(set(full.runtime_state.storage_ptrs()) & set(clone.storage_ptrs()))

    # Verify the only architectural difference disappears when raw gate halves
    # are equal, while preserving paired parameters.
    tied_equal = copy.deepcopy(tied)
    dual_equal = copy.deepcopy(dual)
    _force_equal_raw_gate_halves(tied_equal)
    _force_equal_raw_gate_halves(dual_equal)
    with torch.no_grad():
        tied_out = tied_equal(input_ids).logits
        dual_out = dual_equal(input_ids).logits
    projected_parity = float((tied_out - dual_out).float().abs().max().item())

    sequences = _tiny_sequences(tokenizer)
    tied_batches = cycle_tensor_batches(sequences, batch_size=2, device=device)
    dual_batches = cycle_tensor_batches(sequences, batch_size=2, device=device)
    tied_metrics, tied_optimizer = train_reference_steps(tied, tied_batches, steps=1)
    dual_metrics, dual_optimizer = train_reference_steps(dual, dual_batches, steps=1)
    optimizer_match = compare_optimizer_signatures(tied_optimizer, dual_optimizer)
    run.append(
        "training_metrics.jsonl",
        [
            {"variant": "projected_tied_delta_lm", **tied_metrics[0].to_dict()},
            {"variant": "dual_delta_lm", **dual_metrics[0].to_dict()},
        ],
    )
    gates = [
        _gate("parameter_signature", True, tied.parameter_count(), "identical"),
        _gate("initial_tensor_match", True, True, True),
        _gate("projected_gate_parity", projected_parity <= 1.0e-6, projected_parity, "<=1e-6"),
        _gate("reference_full_chunk", full_chunk_error <= 1.0e-6, full_chunk_error, "<=1e-6"),
        _gate("reference_state_carry", state_error <= 1.0e-6, state_error, "<=1e-6"),
        _gate("state_clone_no_alias", clone_no_alias, clone_no_alias, True),
        _gate(
            "optimizer_state_shape_match", optimizer_match.matched, optimizer_match.matched, True
        ),
        _gate("data_split_disjoint", split_audit["disjoint"], split_audit["overlaps"], []),
        _gate(
            "data_duplicate_free", not split_audit["duplicates"], len(split_audit["duplicates"]), 0
        ),
        _gate(
            "visible_input_audit",
            not split_audit["validation_errors"],
            split_audit["validation_errors"],
            {},
        ),
        _gate(
            "finite_tiny_training", finite_training_metrics(tied_metrics + dual_metrics), True, True
        ),
        _gate(
            "reference_backend_main_blocked", not base.backend_scientific_main_capable, False, False
        ),
    ]
    disposition = (
        "REFERENCE_E26A_CONTRACT_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E26A_CONTRACT_FAIL"
    )
    report = _final_report(
        run, experiment="e26a_operator_data_gate", disposition=disposition, gates=gates
    )
    run.finalize(report, _summary("E26a", disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_tiny_training_stage(
    args: argparse.Namespace,
    raw: dict[str, Any],
    config_path: str,
    *,
    experiment: str,
    steps: int,
) -> int:
    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    base = ModelConfig.tiny_reference()
    tied, dual = build_paired_models(base, seed=26000, device=device)
    run = ArtifactRun(
        experiment=experiment,
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(
        run,
        config_path=config_path,
        raw_config=raw,
        model=dual,
        tokenizer_manifest={
            "id": tokenizer.manifest().tokenizer_id,
            "hash": tokenizer.manifest().manifest_hash,
            "vocab_size": tokenizer.vocab_size,
        },
    )
    sequences = _tiny_sequences(tokenizer)
    tied_metrics, tied_optimizer = train_reference_steps(
        tied,
        cycle_tensor_batches(sequences, batch_size=2, device=device),
        steps=steps,
    )
    dual_metrics, dual_optimizer = train_reference_steps(
        dual,
        cycle_tensor_batches(sequences, batch_size=2, device=device),
        steps=steps,
    )
    rows = [
        {"variant": variant, **metric.to_dict()}
        for variant, metrics in (
            ("projected_tied_delta_lm", tied_metrics),
            ("dual_delta_lm", dual_metrics),
        )
        for metric in metrics
    ]
    run.append("training_metrics.jsonl", rows)
    signatures = compare_optimizer_signatures(tied_optimizer, dual_optimizer)
    gates = [
        _gate("finite_training", finite_training_metrics(tied_metrics + dual_metrics), True, True),
        _gate("optimizer_state_shape_match", signatures.matched, signatures.matched, True),
        _gate(
            "same_steps",
            tied_metrics[-1].step == dual_metrics[-1].step,
            [tied_metrics[-1].step, dual_metrics[-1].step],
            "equal",
        ),
        _gate(
            "same_tokens",
            tied_metrics[-1].tokens_seen == dual_metrics[-1].tokens_seen,
            [tied_metrics[-1].tokens_seen, dual_metrics[-1].tokens_seen],
            "equal",
        ),
    ]
    disposition = (
        f"REFERENCE_{experiment.upper()}_PASS"
        if all(item["passed"] for item in gates)
        else f"REFERENCE_{experiment.upper()}_FAIL"
    )
    report = _final_report(run, experiment=experiment, disposition=disposition, gates=gates)
    run.finalize(report, _summary(experiment, disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_e26d(args: argparse.Namespace, raw: dict[str, Any], config_path: str) -> int:
    run = ArtifactRun(
        experiment="e26d_transaction_eval",
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(run, config_path=config_path, raw_config=raw)
    improvements = {
        26011: {"PRESERVE": 0.0, "ADD": 0.03, "INVALIDATE": 0.025, "SUPERSEDE": 0.001},
        26022: {"PRESERVE": -0.001, "ADD": 0.028, "INVALIDATE": 0.024, "SUPERSEDE": 0.0},
        26033: {"PRESERVE": 0.001, "ADD": 0.032, "INVALIDATE": 0.027, "SUPERSEDE": -0.001},
        26044: {"PRESERVE": 0.0, "ADD": 0.026, "INVALIDATE": 0.023, "SUPERSEDE": 0.001},
        26055: {"PRESERVE": -0.001, "ADD": 0.029, "INVALIDATE": 0.026, "SUPERSEDE": 0.0},
    }
    did = did_by_seed(improvements)
    p_value = exact_sign_flip_pvalue(list(did.values()), alternative="greater")
    interval = bootstrap_interval(list(did.values()), resamples=1000)
    seed_rows = [
        {"seed": seed, "metric": "synthetic_contract_DID", "value": value, "non_evidence": True}
        for seed, value in did.items()
    ]
    run.append("seed_effects.jsonl", seed_rows)
    gates = [
        _gate(
            "did_function",
            len(did) == 5,
            {str(seed): value for seed, value in did.items()},
            "5 seeds",
        ),
        _gate("exact_sign_flip", abs(p_value - 0.03125) < 1.0e-12, p_value, 0.03125),
        _gate(
            "bootstrap_interval_order",
            interval.lower <= interval.estimate <= interval.upper,
            interval.__dict__,
            "lower<=estimate<=upper",
        ),
        _gate("synthetic_values_marked_non_evidence", True, True, True),
    ]
    disposition = (
        "REFERENCE_E26D_STATISTICS_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E26D_STATISTICS_FAIL"
    )
    report = _final_report(
        run, experiment="e26d_transaction_eval", disposition=disposition, gates=gates
    )
    report["note"] = "The numerical effects are synthetic contract fixtures, not model results."
    run.finalize(report, _summary("E26d", disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_e26e(args: argparse.Namespace, raw: dict[str, Any], config_path: str) -> int:
    device = torch.device(args.device)
    config = ModelConfig.tiny_reference("dual_delta_lm")
    model = CatenaLM(config).to(device).eval()
    run = ArtifactRun(
        experiment="e26e_gate_interventions",
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(run, config_path=config_path, raw_config=raw, model=model)
    input_ids = torch.randint(0, config.vocab_size, (1, 19), device=device)
    mask = torch.zeros(19, dtype=torch.bool, device=device)
    mask[5:9] = True
    with torch.no_grad():
        baseline = model(input_ids, return_gate_trace=True)
        tied = model(
            input_ids,
            gate_intervention=GateIntervention(force_tied=True, token_mask=mask),
            return_gate_trace=True,
        )
        erase_zero = model(
            input_ids,
            gate_intervention=GateIntervention(erase_scale=0.0, token_mask=mask),
        )
    tied_change = float((baseline.logits - tied.logits).float().abs().max().item())
    erase_change = float((baseline.logits - erase_zero.logits).float().abs().max().item())
    # Direct hook confinement is identifiable on tokens before the intervention
    # begins. Later tokens may legitimately differ because the intervened state
    # changes subsequent hidden activations.
    prefix = torch.arange(mask.numel(), device=mask.device) < int(mask.nonzero()[0].item())
    outside_equal = True
    for layer, trace in baseline.gate_traces.items():
        tied_trace = tied.gate_traces[layer]
        outside_equal = outside_equal and torch.equal(
            trace.erase[:, prefix], tied_trace.erase[:, prefix]
        )
        outside_equal = outside_equal and torch.equal(
            trace.write[:, prefix], tied_trace.write[:, prefix]
        )
    gates = [
        _gate(
            "intervention_changes_output",
            tied_change > 0 and erase_change > 0,
            [tied_change, erase_change],
            ">0",
        ),
        _gate("non_transaction_gate_confinement", outside_equal, outside_equal, True),
        _gate(
            "gate_trace_available",
            bool(baseline.gate_traces),
            list(baseline.gate_traces),
            "non-empty",
        ),
    ]
    disposition = (
        "REFERENCE_E26E_HOOK_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E26E_HOOK_FAIL"
    )
    report = _final_report(
        run, experiment="e26e_gate_interventions", disposition=disposition, gates=gates
    )
    run.finalize(report, _summary("E26e", disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_e27(args: argparse.Namespace, raw: dict[str, Any], config_path: str) -> int:
    run = ArtifactRun(
        experiment="e27_oracle_decomposition",
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(run, config_path=config_path, raw_config=raw)
    torch.manual_seed(27)
    state = torch.randn(4, 4)
    erase_component = torch.randn(4, 4)
    write_component = torch.randn(4, 4)
    target = state - 0.75 * erase_component + 0.25 * write_component
    fit = fit_bounded_erase_write(state, erase_component, write_component, target, grid_points=101)
    levels = [
        (OracleLevel.METADATA_GATE, 0.50),
        (OracleLevel.METADATA_GATE_ADDRESS, 0.65),
        (OracleLevel.METADATA_GATE_ADDRESS_CANDIDATE, 0.88),
        (OracleLevel.BEHAVIORAL_UPPER_BOUND, 0.94),
    ]
    first = first_substantial_rescue(levels, learned=0.40, exact=0.95)
    e, w = operation_gate_target("INVALIDATE")
    gates = [
        _gate("bounded_fit_erase", abs(fit.erase - 0.75) <= 0.02, fit.erase, "0.75±0.02"),
        _gate("bounded_fit_write", abs(fit.write - 0.25) <= 0.02, fit.write, "0.25±0.02"),
        _gate("operation_metadata_gate", (e, w) == (1.0, 0.0), [e, w], [1.0, 0.0]),
        _gate(
            "first_rescue_classification",
            first == OracleLevel.METADATA_GATE_ADDRESS,
            str(first),
            OracleLevel.METADATA_GATE_ADDRESS.value,
        ),
    ]
    disposition = (
        "REFERENCE_E27_LADDER_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E27_LADDER_FAIL"
    )
    report = _final_report(
        run, experiment="e27_oracle_decomposition", disposition=disposition, gates=gates
    )
    run.finalize(report, _summary("E27", disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_e28(
    args: argparse.Namespace, raw: dict[str, Any], config_path: str, experiment: str
) -> int:
    run = ArtifactRun(
        experiment=experiment,
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(run, config_path=config_path, raw_config=raw)
    torch.manual_seed(28)
    target = torch.nn.functional.normalize(torch.randn(16), dim=0)
    non_targets = torch.nn.functional.normalize(torch.randn(32, 16), dim=-1)
    covariance = active_key_covariance(non_targets)
    direction = covariance_aware_direction(covariance, target, regularization=1.0e-2)
    protected = protected_nullspace_direction(target, non_targets[:8])
    dense = target / torch.dot(target, target)
    dense_spill = worst_non_target_response(dense, non_targets)
    covariance_spill = worst_non_target_response(direction.direction, non_targets)
    gates = [
        _gate(
            "unit_response",
            direction.unit_response_error <= 1.0e-5,
            direction.unit_response_error,
            "<=1e-5",
        ),
        _gate(
            "finite_condition",
            math.isfinite(direction.condition_number),
            direction.condition_number,
            "finite",
        ),
        _gate(
            "covariance_energy_nonnegative",
            direction.covariance_energy >= -1.0e-7,
            direction.covariance_energy,
            ">=0",
        ),
        _gate(
            "protected_projection_unit_response",
            protected.unit_response_error <= 1.0e-5,
            protected.unit_response_error,
            "<=1e-5",
        ),
        _gate(
            "operator_executes",
            math.isfinite(covariance_spill + dense_spill),
            [dense_spill, covariance_spill],
            "finite",
        ),
    ]
    disposition = (
        "REFERENCE_E28_OPERATOR_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E28_OPERATOR_FAIL"
    )
    report = _final_report(run, experiment=experiment, disposition=disposition, gates=gates)
    report["dependency_note"] = (
        "Learned E28 MAIN remains blocked until E27 STRUCTURAL_SPILL_CONFIRMED."
    )
    run.finalize(report, _summary(experiment, disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_e29(
    args: argparse.Namespace, raw: dict[str, Any], config_path: str, experiment: str
) -> int:
    run = ArtifactRun(
        experiment=experiment,
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(run, config_path=config_path, raw_config=raw)
    recurrent = PolicyMeasurement("recurrent_assimilation", 60.0, 0.0, 8.0, 65_536)
    external = PolicyMeasurement("external_read", 0.0, 0.0, 30.0, 262_144)
    full = PolicyMeasurement("full_refresh", 300.0, 0.0, 10.0, 524_288)
    threshold = break_even_queries(recurrent, external)
    frontier = quality_constrained_pareto([recurrent, external, full], queries_per_update=8)
    gates = [
        _gate("finite_break_even", threshold.query_count is not None, threshold.__dict__, "finite"),
        _gate(
            "expected_break_even",
            abs((threshold.query_count or 0.0) - 60.0 / 22.0) < 1.0e-9,
            threshold.query_count,
            60.0 / 22.0,
        ),
        _gate("pareto_nonempty", bool(frontier), [item.policy for item in frontier], "non-empty"),
    ]
    disposition = (
        "REFERENCE_E29_COST_MATH_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E29_COST_MATH_FAIL"
    )
    report = _final_report(run, experiment=experiment, disposition=disposition, gates=gates)
    report["note"] = "Costs are synthetic unit-test fixtures, not hardware measurements."
    run.finalize(report, _summary(experiment, disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


def _run_e30(
    args: argparse.Namespace, raw: dict[str, Any], config_path: str, experiment: str
) -> int:
    run = ArtifactRun(
        experiment=experiment,
        artifact_root=args.artifact_root,
        run_mode="DRY_RUN",
        dry_run=True,
    )
    _base_files(run, config_path=config_path, raw_config=raw)
    gates = [
        _gate("dependency_fail_closed", True, "BLOCKED_WITHOUT_UPSTREAM_REPORT", True),
        _gate(
            "confirmatory_claim_disabled_for_anchor",
            experiment != "e30a_scale_anchor" or raw.get("confirmatory_claim_allowed") is False,
            raw.get("confirmatory_claim_allowed"),
            False,
        ),
    ]
    disposition = (
        "REFERENCE_E30_DEPENDENCY_GUARD_PASS"
        if all(item["passed"] for item in gates)
        else "REFERENCE_E30_DEPENDENCY_GUARD_FAIL"
    )
    report = _final_report(run, experiment=experiment, disposition=disposition, gates=gates)
    report["status"] = "BLOCKED"
    report["allowed_claim"] = "Conditional dependency guard is present."
    run.finalize(report, _summary(experiment, disposition, gates))
    print(run.run_dir)
    return 0 if all(item["passed"] for item in gates) else 1


_RUNNERS: dict[str, Callable[[argparse.Namespace, dict[str, Any], str], int]] = {
    "e26a_operator_data_gate": _run_e26a,
    "e26b_lm_calibration": lambda args, raw, path: _run_tiny_training_stage(
        args, raw, path, experiment="e26b_lm_calibration", steps=max(1, args.steps)
    ),
    "e26c_matched_lm_train": lambda args, raw, path: _run_tiny_training_stage(
        args, raw, path, experiment="e26c_matched_lm_train", steps=max(1, args.steps)
    ),
    "e26d_transaction_eval": _run_e26d,
    "e26e_gate_interventions": _run_e26e,
    "e27_oracle_decomposition": _run_e27,
    "e28a_locality_oracle_pareto": lambda args, raw, path: _run_e28(
        args, raw, path, "e28a_locality_oracle_pareto"
    ),
    "e28b_locality_learned_main": lambda args, raw, path: _run_e28(
        args, raw, path, "e28b_locality_learned_main"
    ),
    "e28c_locality_transfer": lambda args, raw, path: _run_e28(
        args, raw, path, "e28c_locality_transfer"
    ),
    "e29a_policy_correctness": lambda args, raw, path: _run_e29(
        args, raw, path, "e29a_policy_correctness"
    ),
    "e29b_quality_cost_regime": lambda args, raw, path: _run_e29(
        args, raw, path, "e29b_quality_cost_regime"
    ),
    "e30a_scale_anchor": lambda args, raw, path: _run_e30(args, raw, path, "e30a_scale_anchor"),
    "e30b_domain_transfer": lambda args, raw, path: _run_e30(
        args, raw, path, "e30b_domain_transfer"
    ),
    "e30c_final_replication": lambda args, raw, path: _run_e30(
        args, raw, path, "e30c_final_replication"
    ),
}


def run_entrypoint(experiment: str, default_config: str) -> int:
    parser = _parser(experiment, default_config)
    args = parser.parse_args()
    contract, raw = _load_raw_config(args.config)
    if contract.experiment != experiment:
        raise SystemExit(
            f"Config experiment mismatch: entrypoint={experiment}, config={contract.experiment}"
        )
    _guard_mode(args, experiment)
    return _RUNNERS[experiment](args, raw, args.config)
