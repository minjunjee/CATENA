from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from catena.config import load_yaml
from catena.data.render import render_segments
from catena.data.validate import read_chain_jsonl
from catena.methods.encoder_inputs import (
    render_chain_encoder_text,
    render_transaction_encoder_text,
)
from catena.methods.transaction_encoder import EncoderSpec, build_encoder
from catena.models.factory import load_model
from catena.models.hf_stateful import HFStatefulAdapter
from catena.training.encoder_batch import PreparedEncoderInput, prepare_encoder_input
from catena.training.h3_trainer import _as_tensor, _encode_slots, _linear_warmup_decay
from catena.training.losses import categorical_kl, gold_cross_entropy
from catena.training.teacher_cache import load_teacher_scores
from catena.utils.manifest import write_manifest
from catena.utils.seed import seed_everything


def _prepare_paths(model: HFStatefulAdapter, episode, encoder_mode: str):
    sequential = [
        prepare_encoder_input(
            model,
            render_transaction_encoder_text(
                tx,
                closure,
                mode=encoder_mode,
                include_closure=True,
                transaction_index=index,
            ),
        )
        for index, (tx, closure) in enumerate(
            zip(episode.transactions, episode.closures), start=1
        )
    ]
    joint = prepare_encoder_input(
        model,
        render_chain_encoder_text(
            episode,
            mode=encoder_mode,
            include_closure=True,
        ),
    )
    return sequential, joint


def _score_after_prepared_sequence(
    model: HFStatefulAdapter,
    encoder,
    base_state,
    prepared_sequence: list[PreparedEncoderInput],
    query,
    *,
    grad: bool,
):
    import torch

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        slots_sequence = [
            _encode_slots(encoder, prepared).to(
                dtype=model.get_input_embeddings().weight.dtype
            )
            for prepared in prepared_sequence
        ]
        values = []
        for candidate in query.candidates:
            state = model.clone_state(base_state)
            for slots in slots_sequence:
                _, state = model.forward_embeddings(slots, state, grad=grad)
            values.append(
                model.continuation_log_likelihood(
                    state,
                    query.prompt + "\nAnswer:",
                    candidate,
                    grad=grad,
                )
            )
        return torch.stack(values)


def _symmetric_kl(left, right, temperature: float = 1.0):
    return 0.5 * (
        categorical_kl(left, right.detach(), temperature)
        + categorical_kl(right, left.detach(), temperature)
    )


def _select_queries(episode, step: int):
    affected = [q for q in episode.queries if q.kind in {"affected_direct", "affected_derived", "old_rule_probe"}]
    retained = [q for q in episode.queries if q.kind == "unaffected"]
    return affected[step % len(affected)], retained[0]


def train_h4(
    config_path: str,
    *,
    seed: int,
    device: str = "cuda",
    max_steps_override: int | None = None,
    init_checkpoint_override: str | None = None,
) -> Path:
    import torch

    config = load_yaml(config_path)
    seed_everything(seed)
    model = load_model(str(config["model"]), device=device)
    if not isinstance(model, HFStatefulAdapter):
        raise TypeError("H4 training requires the HFStatefulAdapter")
    model.freeze_backbone()

    data_dir = Path(config["data_dir"])
    train_episodes = list(read_chain_jsonl(data_dir / "train.jsonl"))
    val_episodes = list(read_chain_jsonl(data_dir / "val.jsonl"))
    teacher_dir = Path(config["teacher_cache_dir"])
    train_teacher = load_teacher_scores(teacher_dir / "train_teacher_scores.jsonl")
    val_teacher_path = teacher_dir / "val_teacher_scores.jsonl"
    val_teacher = load_teacher_scores(val_teacher_path) if val_teacher_path.exists() else train_teacher

    encoder_config = config["encoder"]
    spec = EncoderSpec(
        input_dim=int(model.get_input_embeddings().weight.shape[1]),
        hidden_dim=int(encoder_config.get("hidden_dim", 512)),
        num_slots=int(encoder_config.get("num_slots", 8)),
        num_layers=int(encoder_config.get("num_layers", 2)),
        num_heads=int(encoder_config.get("num_heads", 8)),
        dropout=float(encoder_config.get("dropout", 0.1)),
    )
    encoder = build_encoder(spec).to(model.device, dtype=torch.float32)
    init_checkpoint = init_checkpoint_override or config.get("init_checkpoint")
    if init_checkpoint:
        payload = torch.load(str(init_checkpoint), map_location="cpu", weights_only=False)
        encoder.load_state_dict(payload["encoder"])
    encoder_mode = str(encoder_config.get("type", "typed_transaction"))

    training = config["training"]
    max_steps = int(max_steps_override or training.get("max_steps", 3000))
    grad_accum = int(training.get("gradient_accumulation", 8))
    eval_every = int(training.get("eval_every_steps", 250))
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = _linear_warmup_decay(
        optimizer,
        warmup_steps=int(training.get("warmup_steps", 100)),
        total_steps=max_steps,
    )
    loss_cfg = config["loss"]
    exact_weight = float(loss_cfg.get("exact_kl", 1.0))
    composition_weight = float(loss_cfg.get("sequential_composed_kl", 0.5))
    retention_weight = float(loss_cfg.get("retention_kl", 0.5))
    gold_weight = float(loss_cfg.get("gold_ce", 0.25))
    temperature = float(loss_cfg.get("temperature", 1.0))

    run_dir = Path(config["output_dir"]) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, {**config, "resolved_seed": seed})
    (run_dir / "encoder_spec.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    log_path = run_dir / "train_log.jsonl"
    rng = random.Random(seed)
    encoder.train()
    optimizer.zero_grad(set_to_none=True)

    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, max_steps + 1):
            episode = train_episodes[rng.randrange(len(train_episodes))]
            teacher = train_teacher[episode.chain_id]
            started = time.perf_counter()
            with torch.no_grad():
                base_state = model.prefill_text(render_segments(episode.history_segments), None)
            sequential_prepared, joint_prepared = _prepare_paths(model, episode, encoder_mode)
            affected, retained = _select_queries(episode, step)

            seq_affected = _score_after_prepared_sequence(
                model, encoder, base_state, sequential_prepared, affected, grad=True
            )
            joint_affected = _score_after_prepared_sequence(
                model, encoder, base_state, [joint_prepared], affected, grad=True
            )
            seq_retained = _score_after_prepared_sequence(
                model, encoder, base_state, sequential_prepared, retained, grad=True
            )
            affected_teacher = _as_tensor(teacher[affected.query_id]["scores"], model.device)
            retained_teacher = _as_tensor(teacher[retained.query_id]["scores"], model.device)

            exact_kl = categorical_kl(seq_affected[None, :], affected_teacher[None, :], temperature)
            retention_kl = categorical_kl(seq_retained[None, :], retained_teacher[None, :], temperature)
            composition_kl = _symmetric_kl(
                seq_affected[None, :], joint_affected[None, :], temperature
            )
            gold_index = torch.tensor([affected.gold_index], dtype=torch.long, device=model.device)
            gold_ce = gold_cross_entropy(seq_affected[None, :], gold_index)
            loss = (
                exact_weight * exact_kl
                + retention_weight * retention_kl
                + composition_weight * composition_kl
                + gold_weight * gold_ce
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite H4 loss at step {step}")
            (loss / grad_accum).backward()

            optimizer_step = step % grad_accum == 0 or step == max_steps
            if optimizer_step:
                torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), float(training.get("gradient_clip_norm", 1.0))
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            row: dict[str, Any] = {
                "step": step,
                "chain_id": episode.chain_id,
                "chain_length": episode.chain_length,
                "loss": float(loss.detach().item()),
                "exact_kl": float(exact_kl.detach().item()),
                "retention_kl": float(retention_kl.detach().item()),
                "composition_kl": float(composition_kl.detach().item()),
                "gold_ce": float(gold_ce.detach().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "step_seconds": time.perf_counter() - started,
            }
            if step % eval_every == 0 or step == max_steps:
                row.update(
                    _quick_validate(
                        model,
                        encoder,
                        val_episodes,
                        val_teacher,
                        encoder_mode,
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
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
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


def _quick_validate(model, encoder, episodes, teacher_scores, encoder_mode: str, max_episodes: int = 16):
    encoder.eval()
    kl_values: list[float] = []
    comp_values: list[float] = []
    for index, episode in enumerate(episodes[:max_episodes]):
        base = model.prefill_text(render_segments(episode.history_segments), None)
        sequential_prepared, joint_prepared = _prepare_paths(model, episode, encoder_mode)
        query, _ = _select_queries(episode, index)
        seq = _score_after_prepared_sequence(
            model, encoder, base, sequential_prepared, query, grad=False
        )
        joint = _score_after_prepared_sequence(
            model, encoder, base, [joint_prepared], query, grad=False
        )
        teacher = _as_tensor(
            teacher_scores[episode.chain_id][query.query_id]["scores"], model.device
        )
        kl_values.append(float(categorical_kl(seq[None, :], teacher[None, :]).item()))
        comp_values.append(float(_symmetric_kl(seq[None, :], joint[None, :]).item()))
    encoder.train()
    return {
        "val_exact_kl": sum(kl_values) / max(1, len(kl_values)),
        "val_composition_kl": sum(comp_values) / max(1, len(comp_values)),
    }
