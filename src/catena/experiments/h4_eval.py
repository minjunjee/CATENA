from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catena.data.render import render_segments
from catena.data.validate import read_chain_jsonl
from catena.eval.metrics import PredictionRecord, aggregate_by, drift_slope, stratified_summary
from catena.experiments.common import categorical_kl_from_scores
from catena.experiments.h3_eval import load_encoder
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.training.h4_trainer import _prepare_paths, _score_after_prepared_sequence
from catena.utils.manifest import write_manifest
from catena.utils.timing import TimingResult, measured


def evaluate_h4(
    *,
    model_config: str,
    checkpoint_path: str,
    data_path: str,
    output_dir: str,
    device: str = "cuda",
    max_episodes: int | None = None,
) -> dict[str, Any]:
    model = load_model(model_config, device=device)
    if not isinstance(model, HFStatefulAdapter):
        raise TypeError("H4 evaluation requires the HFStatefulAdapter")
    model.freeze_backbone()
    encoder, encoder_mode, _ = load_encoder(checkpoint_path, model.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, {
        "model_config": model_config,
        "checkpoint": checkpoint_path,
        "data_path": data_path,
    })

    records: list[PredictionRecord] = []
    path_rows: list[dict[str, Any]] = []
    raw_path = output / "h4_predictions.jsonl"
    with raw_path.open("w", encoding="utf-8") as writer:
        for index, episode in enumerate(read_chain_jsonl(data_path)):
            if max_episodes is not None and index >= max_episodes:
                break
            base = model.prefill_text(render_segments(episode.history_segments), None)
            exact = model.prefill_text(render_segments(episode.refresh_segments), None)
            sequential_prepared, joint_prepared = _prepare_paths(model, episode, encoder_mode)
            for path_name, prepared_sequence in (
                ("sequential", sequential_prepared),
                ("composed", [joint_prepared]),
            ):
                for query in episode.queries:
                    timer = TimingResult()
                    with measured(timer):
                        student = _score_after_prepared_sequence(
                            model,
                            encoder,
                            base,
                            prepared_sequence,
                            query,
                            grad=False,
                        )
                    exact_scores = model.score_candidates(exact, query.prompt, query.candidates)
                    student_list = student.detach().float().cpu().tolist()
                    record = PredictionRecord(
                        episode_id=episode.chain_id,
                        query_id=query.query_id,
                        query_kind=query.kind,
                        policy=path_name,
                        prediction_index=int(student.argmax().item()),
                        gold_index=query.gold_index,
                        exact_prediction_index=exact_scores.prediction_index,
                        teacher_correct=(exact_scores.prediction_index == query.gold_index),
                        logit_kl=categorical_kl_from_scores(exact_scores.log_likelihoods, student_list),
                        latency_ms=timer.milliseconds,
                        state_bytes=model.state_bytes(base),
                        domain=episode.domain,
                        operation="CHAIN",
                        history_tokens=episode.history_token_target,
                        dependency_depth=None,
                        query_gap_tokens=0,
                        chain_length=episode.chain_length,
                    )
                    records.append(record)
                    row = {
                        **asdict(record),
                        "student_scores": student_list,
                        "exact_scores": exact_scores.log_likelihoods,
                    }
                    path_rows.append(row)
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_path = {
        path_name: stratified_summary([r for r in records if r.policy == path_name])
        for path_name in ("sequential", "composed")
    }
    drift: dict[str, float] = {}
    for path_name in ("sequential", "composed"):
        rows = [r for r in records if r.policy == path_name]
        grouped = aggregate_by(rows, "chain_length")
        lengths = sorted(int(k) for k in grouped if k != "None")
        divergences = [grouped[str(length)]["mean_logit_kl"] for length in lengths]
        drift[path_name] = drift_slope(lengths, divergences)
    summary = {"paths": by_path, "drift_slope": drift, "n_records": len(records)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
