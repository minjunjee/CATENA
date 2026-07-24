from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from catena.data.render import render_history_prompt, render_refresh_prompt
from catena.data.validate import read_jsonl
from catena.eval.metrics import PredictionRecord, stratified_summary
from catena.methods.encoder_inputs import render_encoder_text
from catena.methods.transaction_encoder import EncoderSpec, build_encoder
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.training.encoder_batch import prepare_encoder_input
from catena.training.h3_trainer import transport_state
from catena.utils.manifest import write_manifest
from catena.utils.timing import TimingResult, measured


def _softmax(values):
    arr = np.asarray(values, dtype=np.float64)
    arr -= np.max(arr)
    e = np.exp(arr)
    return e / e.sum()


def _kl(p_scores, q_scores):
    p = _softmax(p_scores)
    q = _softmax(q_scores)
    eps = 1e-12
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def load_encoder(checkpoint_path: str | Path, device):
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec_keys = {f.name for f in fields(EncoderSpec)}
    spec = EncoderSpec(**{k: v for k, v in payload["spec"].items() if k in spec_keys})
    encoder = build_encoder(spec).to(device, dtype=torch.float32)
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()
    encoder_config = payload.get("config", {}).get("encoder", {})
    encoder_mode = str(encoder_config.get("type", "typed_transaction"))
    include_closure = bool(encoder_config.get("include_closure", True))
    return encoder, encoder_mode, include_closure


def evaluate_h3(
    *,
    model_config: str,
    checkpoint_path: str,
    data_path: str,
    output_dir: str,
    device: str = "cuda",
    max_episodes: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    model = load_model(model_config, device=device)
    if not isinstance(model, HFStatefulAdapter):
        raise TypeError("H3 evaluation requires the HFStatefulAdapter")
    model.freeze_backbone()
    encoder, encoder_mode, include_closure = load_encoder(checkpoint_path, model.device)
    output = Path(output_dir)
    if num_shards > 1:
        output = output / f"shard_{shard_index:02d}_of_{num_shards:02d}"
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output,
        {
            "model_config": model_config,
            "checkpoint": checkpoint_path,
            "data_path": data_path,
            "shard_index": shard_index,
            "num_shards": num_shards,
        },
    )
    records: list[PredictionRecord] = []
    raw_path = output / "catena_predictions.jsonl"
    with raw_path.open("w", encoding="utf-8") as writer:
        processed = 0
        for episode_index, episode in enumerate(read_jsonl(data_path)):
            if episode_index % num_shards != shard_index:
                continue
            if max_episodes is not None and processed >= max_episodes:
                break
            base_state = model.prefill_text(render_history_prompt(episode), None)
            exact_state = model.prefill_text(render_refresh_prompt(episode), None)
            rendered = render_encoder_text(
                episode, mode=encoder_mode, include_closure=include_closure
            )
            prepared = prepare_encoder_input(model, rendered)
            timer = TimingResult()
            with measured(timer):
                transported_state, _ = transport_state(
                    model, encoder, base_state, prepared, grad=False
                )
            for query in episode.queries:
                student_scores = model.score_candidates(
                    transported_state, query.prompt, query.candidates
                )
                exact = model.score_candidates(
                    exact_state, query.prompt, query.candidates
                )
                prediction = int(student_scores.prediction_index)
                record = PredictionRecord(
                    episode_id=episode.episode_id,
                    query_id=query.query_id,
                    query_kind=query.kind,
                    policy=f"catena:{encoder_mode}",
                    prediction_index=prediction,
                    gold_index=query.gold_index,
                    exact_prediction_index=exact.prediction_index,
                    teacher_correct=(exact.prediction_index == query.gold_index),
                    logit_kl=_kl(
                        exact.log_likelihoods,
                        student_scores.log_likelihoods,
                    ),
                    latency_ms=timer.milliseconds,
                    state_bytes=model.state_bytes(transported_state),
                    domain=episode.domain,
                    operation=episode.transaction.operation,
                    history_tokens=episode.history_token_target,
                    dependency_depth=episode.dependency_depth,
                    query_gap_tokens=episode.query_gap_tokens,
                )
                records.append(record)
                writer.write(
                    json.dumps(
                        {
                            **asdict(record),
                            "student_scores": student_scores.log_likelihoods,
                            "exact_scores": exact.log_likelihoods,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            processed += 1
    summary = {
        "episodes": processed,
        "encoder_mode": encoder_mode,
        "include_closure": include_closure,
        "metrics": stratified_summary(records),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
