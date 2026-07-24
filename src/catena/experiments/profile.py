from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from catena.config import load_yaml
from catena.data.generator import generate_episode
from catena.data.render import render_history_prompt, render_typed_closure
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.models.rwkv_pip import RWKVPipAdapter
from catena.utils.manifest import write_manifest
from catena.utils.timing import synchronize_if_cuda


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[index])


def _decode_profile(model, state, prompt: str, max_new_tokens: int) -> tuple[float, float]:
    """Return time-to-first-token and remaining greedy-decode latency in ms."""
    if isinstance(model, HFStatefulAdapter):
        import torch

        prompt_ids = model.encode(prompt)
        if not prompt_ids:
            prompt_ids = [model.tokenizer.eos_token_id]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        synchronize_if_cuda()
        started = time.perf_counter()
        outputs, current = model._forward(input_ids=ids, state=model.clone_state(state), grad=False)
        token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        synchronize_if_cuda()
        ttfa = (time.perf_counter() - started) * 1000.0
        synchronize_if_cuda()
        started = time.perf_counter()
        for _ in range(max(0, max_new_tokens - 1)):
            token = torch.tensor([[token_id]], dtype=torch.long, device=model.device)
            outputs, current = model._forward(input_ids=token, state=current, grad=False)
            token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        synchronize_if_cuda()
        decode = (time.perf_counter() - started) * 1000.0
        return ttfa, decode

    if isinstance(model, RWKVPipAdapter):
        import copy
        import torch

        prompt_ids = model.encode(prompt)
        synchronize_if_cuda()
        started = time.perf_counter()
        logits, current = model.model.forward(prompt_ids, copy.deepcopy(state))
        token_id = int(torch.argmax(logits).item())
        synchronize_if_cuda()
        ttfa = (time.perf_counter() - started) * 1000.0
        synchronize_if_cuda()
        started = time.perf_counter()
        for _ in range(max(0, max_new_tokens - 1)):
            logits, current = model.model.forward([token_id], current)
            token_id = int(torch.argmax(logits).item())
        synchronize_if_cuda()
        decode = (time.perf_counter() - started) * 1000.0
        return ttfa, decode
    raise TypeError(type(model))


def run_profile(config_path: str, *, model_index: int, device: str = "cuda") -> dict[str, Any]:
    import torch

    config = load_yaml(config_path)
    model_configs = list(config["models"])
    if not 0 <= model_index < len(model_configs):
        raise ValueError("model_index is out of range")
    model_config = str(model_configs[model_index])
    model = load_model(model_config, device=device)
    output = Path(config["output_dir"]) / f"model_{model_index}"
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, {**config, "resolved_model": model_config})

    repetitions = int(config.get("repetitions", 20))
    warmups = int(config.get("warmup_repetitions", 3))
    generation_tokens = int(config.get("generation_tokens", 32))
    rows: list[dict[str, Any]] = []

    for history_length in config["history_lengths"]:
        episode = generate_episode(
            split="profile",
            index=int(history_length),
            seed=20260723,
            history_token_target=int(history_length),
            domain="api",
            operation="SUPERSEDE",
            dependency_depth=2,
            query_gap_tokens=0,
            schema_family="payment-client",
        )
        history = render_history_prompt(episode)
        patch = render_typed_closure(episode.transaction, episode.closure)
        prompt = episode.queries[1].prompt + "\nAnswer:"
        prefill_values: list[float] = []
        update_values: list[float] = []
        ttfa_values: list[float] = []
        decode_values: list[float] = []
        state_bytes = 0
        peak_values: list[int] = []

        for repetition in range(warmups + repetitions):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            synchronize_if_cuda()
            started = time.perf_counter()
            base = model.prefill_text(history, None)
            synchronize_if_cuda()
            prefill_ms = (time.perf_counter() - started) * 1000.0

            synchronize_if_cuda()
            started = time.perf_counter()
            updated = model.prefill_text(patch, model.clone_state(base))
            synchronize_if_cuda()
            update_ms = (time.perf_counter() - started) * 1000.0
            ttfa_ms, decode_ms = _decode_profile(model, updated, prompt, generation_tokens)
            state_bytes = model.state_bytes(updated)
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            if repetition >= warmups:
                prefill_values.append(prefill_ms)
                update_values.append(update_ms)
                ttfa_values.append(ttfa_ms)
                decode_values.append(decode_ms)
                peak_values.append(peak)

        rows.append(
            {
                "model_config": model_config,
                "history_tokens_target": int(history_length),
                "prefill_ms_median": statistics.median(prefill_values),
                "prefill_ms_p95": _percentile(prefill_values, 0.95),
                "update_ms_median": statistics.median(update_values),
                "update_ms_p95": _percentile(update_values, 0.95),
                "ttfa_ms_median": statistics.median(ttfa_values),
                "decode_32_ms_median": statistics.median(decode_values),
                "resident_state_bytes": state_bytes,
                "peak_allocated_bytes_median": int(statistics.median(peak_values)),
                "repetitions": repetitions,
            }
        )

    payload = {"model_config": model_config, "rows": rows}
    (output / "profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
