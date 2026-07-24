from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from catena.config import load_yaml
from catena.data.generator import generate_episode
from catena.data.render import render_history_prompt, render_refresh_prompt
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.utils.manifest import write_manifest


def _score_vector(model, state, prompt: str, candidates: list[str]) -> np.ndarray:
    scores = model.score_candidates(state, prompt, candidates)
    return np.asarray(scores.log_likelihoods, dtype=np.float64)


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "passed": False,
            "reason": f"shape mismatch: {left.shape} vs {right.shape}",
        }
    diff = np.abs(left - right)
    return {
        "passed": bool(np.all(np.isfinite(diff))),
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "ranking_agreement": bool(int(left.argmax()) == int(right.argmax())),
        "left": left.tolist(),
        "right": right.tolist(),
    }


def _hf_gates(model: HFStatefulAdapter, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    episode = generate_episode(
        split="runtime",
        index=0,
        seed=20260723,
        history_token_target=768,
        domain="api",
        operation="SUPERSEDE",
        dependency_depth=2,
        query_gap_tokens=64,
        schema_family="payment-client",
    )
    query = episode.queries[1]
    prompt = query.prompt
    candidates = list(query.candidates)
    history = render_history_prompt(episode)
    ids = model.encode(history)
    if len(ids) < 8:
        raise RuntimeError("Runtime-gate history tokenization is unexpectedly short")

    result: dict[str, Any] = {
        "history_tokens": len(ids),
        "model_class": type(model.model).__name__,
        "device": str(model.device),
        "dtype": str(model.get_input_embeddings().weight.dtype),
    }

    # 1) Full prefill versus token-exact chunked prefill.
    full = model.prefill_token_ids(ids, None, grad=False)
    chunked = None
    for start in range(0, len(ids), 97):
        chunked = model.prefill_token_ids(ids[start : start + 97], chunked, grad=False)
    assert chunked is not None
    full_scores = _score_vector(model, full, prompt, candidates)
    chunk_scores = _score_vector(model, chunked, prompt, candidates)
    cmp_full_chunk = _comparison(full_scores, chunk_scores)
    cmp_full_chunk["passed"] = bool(
        cmp_full_chunk["ranking_agreement"]
        and cmp_full_chunk["max_abs_error"] <= atol + rtol * max(1.0, float(np.abs(full_scores).max()))
    )
    result["full_vs_chunked"] = cmp_full_chunk

    # 2) Token IDs versus their exact input embeddings.
    short_ids = ids[: min(64, len(ids))]
    token_state = model.prefill_token_ids(short_ids, None, grad=False)
    token_tensor = torch.tensor([short_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        embeds = model.get_input_embeddings()(token_tensor)
    embed_state = model.prefill_embeddings(embeds, None, grad=False)
    token_scores = _score_vector(model, token_state, prompt, candidates)
    embed_scores = _score_vector(model, embed_state, prompt, candidates)
    cmp_token_embed = _comparison(token_scores, embed_scores)
    cmp_token_embed["passed"] = bool(
        cmp_token_embed["ranking_agreement"]
        and cmp_token_embed["max_abs_error"] <= atol + rtol * max(1.0, float(np.abs(token_scores).max()))
    )
    result["token_vs_embedding"] = cmp_token_embed

    # 3) Cloning must not alias the original cache.
    original_before = _score_vector(model, full, prompt, candidates)
    clone = model.clone_state(full)
    _ = model.prefill_text("\nIndependent branch mutation.", clone)
    original_after = _score_vector(model, full, prompt, candidates)
    cmp_clone = _comparison(original_before, original_after)
    cmp_clone["passed"] = bool(
        cmp_clone["ranking_agreement"]
        and cmp_clone["max_abs_error"] <= atol + rtol * max(1.0, float(np.abs(original_before).max()))
    )
    result["clone_no_alias"] = cmp_clone

    # 4) A continuous slot must carry gradient through the native model path.
    model.freeze_backbone()
    slot_source = model.get_input_embeddings()(token_tensor[:, :4]).detach().float()
    slot_source.requires_grad_(True)
    slot_dtype = model.get_input_embeddings().weight.dtype
    _, transported = model.forward_embeddings(slot_source.to(slot_dtype), None, grad=True)
    ll = model.continuation_log_likelihood(
        transported,
        "Current answer:\nAnswer:",
        candidates[0],
        grad=True,
    )
    (-ll).backward()
    grad = slot_source.grad
    grad_ok = grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0.0
    result["embedding_gradient"] = {
        "passed": grad_ok,
        "loss": float((-ll).detach().item()),
        "grad_l1": None if grad is None else float(grad.abs().sum().item()),
        "grad_max": None if grad is None else float(grad.abs().max().item()),
    }

    # 5) Transformer cache crop/re-prefill parity.  Recurrent caches may not expose crop.
    crop_result: dict[str, Any]
    split = len(ids) // 2
    try:
        cropped = model.crop_state(full, split)
        repaired = model.prefill_token_ids(ids[split:], cropped, grad=False)
        repaired_scores = _score_vector(model, repaired, prompt, candidates)
        crop_result = _comparison(full_scores, repaired_scores)
        crop_result["passed"] = bool(
            crop_result["ranking_agreement"]
            and crop_result["max_abs_error"] <= atol + rtol * max(1.0, float(np.abs(full_scores).max()))
        )
        crop_result["prefix_tokens"] = split
    except (NotImplementedError, AttributeError, TypeError, RuntimeError) as exc:
        crop_result = {"passed": None, "skipped": True, "reason": repr(exc)}
    result["cache_crop_reprefill"] = crop_result

    # 6) Serialization is an optimization gate, not a scientific hard requirement.
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(full, path)
            restored = torch.load(path, map_location=model.device, weights_only=False)
            restored_scores = _score_vector(model, restored, prompt, candidates)
        ser = _comparison(full_scores, restored_scores)
        ser["passed"] = bool(ser["ranking_agreement"])
    except Exception as exc:  # cache objects are backend-version specific
        ser = {"passed": None, "skipped": True, "reason": repr(exc)}
    result["state_serialize_restore_optional"] = ser

    # 7) Exact-refresh path must execute and produce finite scores.
    exact = model.prefill_text(render_refresh_prompt(episode), None)
    exact_scores = _score_vector(model, exact, prompt, candidates)
    result["exact_refresh_smoke"] = {
        "passed": bool(np.all(np.isfinite(exact_scores))),
        "prediction_index": int(exact_scores.argmax()),
        "gold_index": int(query.gold_index),
        "state_bytes": int(model.state_bytes(exact)),
    }
    result["state_bytes"] = int(model.state_bytes(full))
    return result


def _generic_smoke(model) -> dict[str, Any]:
    episode = generate_episode(
        split="runtime",
        index=1,
        seed=20260723,
        history_token_target=256,
        domain="workflow",
        operation="AMEND",
        dependency_depth=1,
        query_gap_tokens=0,
        schema_family="release-flow",
    )
    state = model.prefill_text(render_history_prompt(episode), None)
    query = episode.queries[0]
    scores = model.score_candidates(state, query.prompt, query.candidates)
    return {
        "load_and_score": {
            "passed": all(math.isfinite(v) for v in scores.log_likelihoods),
            "prediction_index": scores.prediction_index,
            "gold_index": query.gold_index,
        },
        "state_bytes": int(model.state_bytes(state)),
        "skipped_checks": [
            "token_vs_embedding",
            "embedding_gradient",
            "cache_crop_reprefill",
        ],
    }


def run_runtime_gates(
    config_path: str,
    *,
    model_index: int,
    device: str = "cuda",
) -> dict[str, Any]:
    config = load_yaml(config_path)
    models = list(config["models"])
    if not 0 <= model_index < len(models):
        raise ValueError(f"model_index must be in [0, {len(models) - 1}]")
    model_config = str(models[model_index])
    started = time.perf_counter()
    model = load_model(model_config, device=device)
    if isinstance(model, HFStatefulAdapter):
        checks = _hf_gates(
            model,
            atol=float(config.get("atol", 1e-3)),
            rtol=float(config.get("rtol", 1e-3)),
        )
        hard_names = [
            "full_vs_chunked",
            "token_vs_embedding",
            "clone_no_alias",
            "embedding_gradient",
            "exact_refresh_smoke",
        ]
    else:
        checks = _generic_smoke(model)
        hard_names = ["load_and_score"]

    failures = [name for name in hard_names if checks.get(name, {}).get("passed") is not True]
    payload = {
        "experiment": str(config.get("experiment", "e01_runtime")),
        "model_index": model_index,
        "model_config": model_config,
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "hard_failures": failures,
        "passed": not failures,
        "checks": checks,
    }
    output = Path(config["output_dir"]) / f"model_{model_index}"
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, {**config, "resolved_model": model_config, "model_index": model_index})
    (output / "runtime_gates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"Runtime hard gate failed for {model_config}: {failures}")
    return payload
