from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from catena.config import load_yaml
from catena.data.render import render_history_prompt
from catena.data.validate import read_jsonl
from catena.experiments.h3_eval import load_encoder
from catena.methods.encoder_inputs import render_encoder_text
from catena.methods.policies import apply_text_policy
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.training.encoder_batch import prepare_encoder_input
from catena.training.h3_trainer import _encode_slots
from catena.utils.manifest import write_manifest


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                    return payload if isinstance(payload, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _naturalize_prompt(prompt: str, episode_index: int) -> str:
    variants = [
        "Using only the latest valid configuration, emit the executable JSON action.",
        "The environment has changed. Produce the JSON call that is valid now.",
        "Ignore superseded rules and return the current tool invocation as JSON.",
        "Resolve the current version and output one JSON action with no explanation.",
    ]
    # Retain the schema-family identifier from the original query while varying the
    # instruction surface form.
    match = re.search(r"for ([\w-]+)", prompt)
    family = match.group(1) if match else "the active configuration"
    return f"{variants[episode_index % len(variants)]} Target: {family}."


def _catena_state(model, episode, base_state, checkpoint: str):
    import torch

    if not isinstance(model, HFStatefulAdapter):
        raise TypeError("Learned CATENA generation requires the HFStatefulAdapter")
    encoder, mode, include_closure = load_encoder(checkpoint, model.device)
    rendered = render_encoder_text(episode, mode=mode, include_closure=include_closure)
    prepared = prepare_encoder_input(model, rendered)
    with torch.no_grad():
        slots = _encode_slots(encoder, prepared).to(
            dtype=model.get_input_embeddings().weight.dtype
        )
        return model.prefill_embeddings(slots, model.clone_state(base_state), grad=False)


def run_toolcall_eval(
    config_path: str,
    *,
    run_index: int,
    device: str = "cuda",
    max_episodes: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    run = list(config["runs"])[run_index]
    model = load_model(str(run["model"]), device=device)
    policy = str(run["policy"])
    checkpoint = run.get("checkpoint")
    output = Path(config["output_dir"]) / f"run_{run_index}_{policy}"
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, {**config, "resolved_run": run})

    counts = {
        "episodes": 0,
        "schema_valid": 0,
        "tool_name_exact": 0,
        "argument_exact": 0,
        "simulator_success": 0,
        "stale_action": 0,
    }
    raw_path = output / "generations.jsonl"
    with raw_path.open("w", encoding="utf-8") as writer:
        for episode_index, episode in enumerate(
            read_jsonl(Path(config["data_dir"]) / "test.jsonl")
        ):
            limit = max_episodes or int(config.get("max_episodes", 300))
            if counts["episodes"] >= limit:
                break
            query = next(q for q in episode.queries if q.kind == "affected_derived")
            base = model.prefill_text(render_history_prompt(episode), None)
            if policy == "catena":
                if not checkpoint:
                    raise ValueError("CATENA run requires a checkpoint")
                state = _catena_state(model, episode, base, str(checkpoint))
            else:
                state = apply_text_policy(model, episode, base, policy)
            prompt = _naturalize_prompt(query.prompt, episode_index) + "\nJSON:"
            token_ids, _ = model.generate_greedy(
                state,
                prompt,
                max_new_tokens=int(config.get("max_new_tokens", 96)),
            )
            generated_text = model.decode(token_ids)
            parsed = _extract_json(generated_text)
            gold = json.loads(query.gold)
            old_action = json.loads(str(episode.metadata["old_action"]))
            counts["episodes"] += 1
            if parsed is not None:
                counts["schema_valid"] += 1
                counts["tool_name_exact"] += int(parsed.get("tool") == gold.get("tool"))
                gold_args = {k: v for k, v in gold.items() if k != "tool"}
                parsed_args = {k: v for k, v in parsed.items() if k != "tool"}
                counts["argument_exact"] += int(parsed_args == gold_args)
                counts["simulator_success"] += int(parsed == gold)
                counts["stale_action"] += int(parsed == old_action)
            writer.write(
                json.dumps(
                    {
                        "episode_id": episode.episode_id,
                        "policy": policy,
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "parsed": parsed,
                        "gold": gold,
                        "old_action": old_action,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    n = max(1, counts["episodes"])
    metrics = {
        "episodes": counts["episodes"],
        "schema_validity": counts["schema_valid"] / n,
        "tool_name_exact": counts["tool_name_exact"] / n,
        "argument_exact_match": counts["argument_exact"] / n,
        "simulator_success": counts["simulator_success"] / n,
        "stale_field_rate": counts["stale_action"] / n,
    }
    (output / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
