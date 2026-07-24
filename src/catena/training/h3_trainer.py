from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catena.config import load_yaml
from catena.data.render import render_history_prompt
from catena.data.validate import read_jsonl
from catena.methods.encoder_inputs import render_encoder_text
from catena.methods.transaction_encoder import EncoderSpec, build_encoder
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.training.encoder_batch import PreparedEncoderInput, prepare_encoder_input
from catena.training.losses import categorical_kl, gold_cross_entropy
from catena.training.teacher_cache import load_teacher_scores
from catena.utils.manifest import write_manifest
from catena.utils.seed import seed_everything


def _as_tensor(values: list[float], device):
    import torch

    return torch.tensor(values, dtype=torch.float32, device=device)


def _encode_slots(encoder, prepared: PreparedEncoderInput):
    # Keep the small encoder in fp32 for optimizer stability.  The resulting slots
    # are converted to the frozen backbone's embedding dtype before native forward.
    embeddings = prepared.embeddings.float()
    return encoder(
        embeddings,
        attention_mask=prepared.attention_mask,
        field_type_ids=prepared.field_type_ids,
    )


def transport_state(
    model: HFStatefulAdapter,
    encoder,
    base_state,
    prepared: PreparedEncoderInput,
    *,
    grad: bool = False,
):
    """Apply the encoder once and return the transported model state.

    This helper is used for evaluation and profiling so update latency contains
    only encoder + K-slot native forward, not candidate scoring.
    """
    import torch

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        slots = _encode_slots(encoder, prepared)
        slots = slots.to(dtype=model.get_input_embeddings().weight.dtype)
        _, transported = model.forward_embeddings(
            slots, model.clone_state(base_state), grad=grad
        )
    return transported, slots


def _student_candidate_scores(
    model: HFStatefulAdapter,
    encoder,
    base_state,
    prepared: PreparedEncoderInput,
    query,
    *,
    grad: bool = True,
):
    """Score all candidates after one transaction-conditioned state transport.

    The encoder is evaluated once.  The native model transition is repeated from a
    clean clone of the same base state for each candidate because Hugging Face cache
    objects may update in place.  Gradients flow only through the encoder and the
    continuous transaction slots; the language-model backbone remains frozen.
    """

    import torch

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        slots = _encode_slots(encoder, prepared)
        target_dtype = model.get_input_embeddings().weight.dtype
        slots = slots.to(dtype=target_dtype)
        values = []
        for candidate in query.candidates:
            state = model.clone_state(base_state)
            _, transported = model.forward_embeddings(slots, state, grad=grad)
            ll = model.continuation_log_likelihood(
                transported,
                query.prompt + "\nAnswer:",
                candidate,
                grad=grad,
            )
            values.append(ll)
        return torch.stack(values)


def _prepare_transaction(
    model: HFStatefulAdapter,
    episode,
    encoder_mode: str,
    *,
    include_closure: bool = True,
):
    rendered = render_encoder_text(
        episode,
        mode=encoder_mode,
        include_closure=include_closure,
    )
    return prepare_encoder_input(model, rendered)


def _select_queries(episode, step: int):
    affected = [
        q
        for q in episode.queries
        if q.kind in {"affected_direct", "affected_derived", "old_rule_probe"}
    ]
    retained = [q for q in episode.queries if q.kind == "unaffected"]
    affected_query = affected[step % len(affected)]
    retained_query = retained[0]
    return affected_query, retained_query


def _validate(
    *,
    model: HFStatefulAdapter,
    encoder,
    episodes,
    teacher_scores,
    encoder_mode: str,
    include_closure: bool,
    max_episodes: int = 32,
) -> dict[str, float]:
    encoder.eval()
    correct = 0
    oracle = 0
    total = 0
    losses: list[float] = []
    for index, episode in enumerate(episodes[:max_episodes]):
        base = model.prefill_text(render_history_prompt(episode), None)
        prepared = _prepare_transaction(
            model, episode, encoder_mode, include_closure=include_closure
        )
        affected, retained = _select_queries(episode, index)
        for query in (affected, retained):
            student = _student_candidate_scores(
                model,
                encoder,
                base,
                prepared,
                query,
                grad=False,
            )
            teacher = _as_tensor(
                teacher_scores[episode.episode_id][query.query_id]["scores"], model.device
            )
            losses.append(float(categorical_kl(student[None, :], teacher[None, :]).item()))
            prediction = int(student.argmax().item())
            correct += int(prediction == query.gold_index)
            oracle += int(prediction == int(teacher.argmax().item()))
            total += 1
    encoder.train()
    return {
        "val_candidate_accuracy": correct / max(1, total),
        "val_oracle_agreement": oracle / max(1, total),
        "val_kl": sum(losses) / max(1, len(losses)),
    }


def _linear_warmup_decay(optimizer, *, warmup_steps: int, total_steps: int):
    import torch

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_h3(
    config_path: str,
    *,
    seed: int,
    device: str = "cuda",
    max_steps_override: int | None = None,
) -> Path:
    import torch

    config = load_yaml(config_path)
    seed_everything(seed)
    model = load_model(str(config["model"]), device=device)
    if not isinstance(model, HFStatefulAdapter):
        raise TypeError("H3 training requires the HFStatefulAdapter")
    model.freeze_backbone()

    teacher_dir = Path(config["teacher_cache_dir"])
    teacher_scores = load_teacher_scores(teacher_dir / "train_teacher_scores.jsonl")
    val_teacher_path = teacher_dir / "val_teacher_scores.jsonl"
    val_teacher_scores = (
        load_teacher_scores(val_teacher_path) if val_teacher_path.exists() else teacher_scores
    )

    data_dir = Path(config.get("data_dir", "data/processed/main"))
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"Training data not found under {data_dir}")
    train_episodes = list(read_jsonl(data_dir / "train.jsonl"))
    val_episodes = list(read_jsonl(data_dir / "val.jsonl"))
    if not train_episodes:
        raise RuntimeError(f"No training episodes found under {data_dir}")

    embedding_dim = int(model.get_input_embeddings().weight.shape[1])
    encoder_config = config["encoder"]
    spec = EncoderSpec(
        input_dim=embedding_dim,
        hidden_dim=int(encoder_config.get("hidden_dim", 512)),
        num_slots=int(encoder_config.get("num_slots", 8)),
        num_layers=int(encoder_config.get("num_layers", 2)),
        num_heads=int(encoder_config.get("num_heads", 8)),
        dropout=float(encoder_config.get("dropout", 0.1)),
    )
    encoder = build_encoder(spec).to(model.device, dtype=torch.float32)
    encoder_mode = str(encoder_config.get("type", "typed_transaction"))
    include_closure = bool(encoder_config.get("include_closure", True))

    training = config["training"]
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    max_steps = int(max_steps_override or training.get("max_steps", 6000))
    grad_accum = int(training.get("gradient_accumulation", 16))
    eval_every = int(training.get("eval_every_steps", 500))
    grad_clip = float(training.get("gradient_clip_norm", 1.0))
    scheduler = _linear_warmup_decay(
        optimizer,
        warmup_steps=int(training.get("warmup_steps", 200)),
        total_steps=max_steps,
    )

    loss_cfg = config["loss"]
    lambda_affected = float(loss_cfg.get("affected_kl", 1.0))
    lambda_retention = float(loss_cfg.get("retention_kl", 0.6))
    lambda_gold = float(loss_cfg.get("gold_ce", 0.5))
    temperature = float(loss_cfg.get("temperature", 1.0))

    run_dir = Path(config["output_dir"]) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, {**config, "resolved_seed": seed})
    (run_dir / "encoder_spec.json").write_text(
        json.dumps(asdict(spec), indent=2), encoding="utf-8"
    )
    log_path = run_dir / "train_log.jsonl"
    rng = random.Random(seed)
    optimizer.zero_grad(set_to_none=True)
    encoder.train()

    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, max_steps + 1):
            episode = train_episodes[rng.randrange(len(train_episodes))]
            teacher = teacher_scores.get(episode.episode_id)
            if teacher is None:
                raise KeyError(f"Teacher cache is missing episode {episode.episode_id}")
            started = time.perf_counter()
            with torch.no_grad():
                base_state = model.prefill_text(render_history_prompt(episode), None)
            prepared = _prepare_transaction(
                model, episode, encoder_mode, include_closure=include_closure
            )
            affected_query, retained_query = _select_queries(episode, step)

            affected_student = _student_candidate_scores(
                model, encoder, base_state, prepared, affected_query, grad=True
            )
            retention_student = _student_candidate_scores(
                model, encoder, base_state, prepared, retained_query, grad=True
            )
            affected_teacher = _as_tensor(
                teacher[affected_query.query_id]["scores"], model.device
            )
            retention_teacher = _as_tensor(
                teacher[retained_query.query_id]["scores"], model.device
            )
            affected_gold = torch.tensor(
                [affected_query.gold_index], dtype=torch.long, device=model.device
            )
            retained_gold = torch.tensor(
                [retained_query.gold_index], dtype=torch.long, device=model.device
            )

            affected_kl = categorical_kl(
                affected_student[None, :], affected_teacher[None, :], temperature
            )
            retention_kl = categorical_kl(
                retention_student[None, :], retention_teacher[None, :], temperature
            )
            gold_ce = 0.5 * (
                gold_cross_entropy(affected_student[None, :], affected_gold)
                + gold_cross_entropy(retention_student[None, :], retained_gold)
            )
            loss = (
                lambda_affected * affected_kl
                + lambda_retention * retention_kl
                + lambda_gold * gold_ce
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {float(loss.detach().item())}"
                )
            (loss / grad_accum).backward()

            optimizer_step = step % grad_accum == 0 or step == max_steps
            grad_norm = float("nan")
            if optimizer_step:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                grad_norm = float(grad_norm_tensor.detach().item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            payload: dict[str, Any] = {
                "step": step,
                "episode_id": episode.episode_id,
                "encoder_mode": encoder_mode,
                "include_closure": include_closure,
                "loss": float(loss.detach().item()),
                "affected_kl": float(affected_kl.detach().item()),
                "retention_kl": float(retention_kl.detach().item()),
                "gold_ce": float(gold_ce.detach().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm,
                "optimizer_step": optimizer_step,
                "step_seconds": time.perf_counter() - started,
            }
            if step % eval_every == 0 or step == max_steps:
                payload.update(
                    _validate(
                        model=model,
                        encoder=encoder,
                        episodes=val_episodes,
                        teacher_scores=val_teacher_scores,
                        encoder_mode=encoder_mode,
                        include_closure=include_closure,
                    )
                )
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "spec": asdict(spec),
                        "seed": seed,
                        "step": step,
                        "config": config,
                    },
                    run_dir / f"checkpoint_step_{step}.pt",
                )
            log.write(json.dumps(payload, ensure_ascii=False) + "\n")
            log.flush()

    final_path = run_dir / "encoder_final.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "spec": asdict(spec),
            "seed": seed,
            "step": max_steps,
            "config": config,
        },
        final_path,
    )
    return final_path
