from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from catena.config import load_yaml

app = typer.Typer(no_args_is_help=True, help="CATENA research experiment CLI")
console = Console()


@app.command("audit")
def audit() -> None:
    """Print a concise Python/PyTorch GPU audit."""
    payload: dict[str, object] = {}
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            }
        )
        if torch.cuda.is_available():
            payload["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        payload["error"] = repr(exc)
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("smoke")
def smoke() -> None:
    """Run a CPU-only end-to-end smoke test with the mock model."""
    from catena.data.generator import generate_episode
    from catena.methods.policies import apply_text_policy, build_base_state
    from catena.models.mock import MockStatefulModel

    episode = generate_episode(
        split="smoke",
        index=0,
        seed=13,
        history_token_target=128,
        domain="api",
        operation="SUPERSEDE",
        dependency_depth=1,
        query_gap_tokens=0,
        schema_family="payment-client",
    )
    model = MockStatefulModel()
    base = build_base_state(model, episode)
    exact = apply_text_policy(model, episode, base, "exact_refresh")
    typed = apply_text_policy(model, episode, base, "typed_closure")
    query = episode.queries[0]
    result = {
        "episode_id": episode.episode_id,
        "base_bytes": model.state_bytes(base),
        "exact_prediction": model.score_candidates(exact, query.prompt, query.candidates).prediction_index,
        "typed_prediction": model.score_candidates(typed, query.prompt, query.candidates).prediction_index,
        "gold_index": query.gold_index,
    }
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("data-generate")
def data_generate(config: str = typer.Option(..., help="YAML data configuration")) -> None:
    from catena.data.generator import generate_from_config

    manifest = generate_from_config(config)
    console.print_json(json.dumps(manifest, ensure_ascii=False))


@app.command("data-generate-chains")
def data_generate_chains(config: str = typer.Option(..., help="YAML data configuration")) -> None:
    from catena.data.chain_generator import generate_chains_from_config

    manifest = generate_chains_from_config(config)
    console.print_json(json.dumps(manifest, ensure_ascii=False))


@app.command("data-validate")
def data_validate(path: str = typer.Option(..., help="JSONL file")) -> None:
    from catena.data.validate import validate_file

    report = validate_file(path)
    console.print_json(json.dumps(report, ensure_ascii=False))
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command("config-audit")
def config_audit(
    root: str = typer.Option(".", help="Repository root"),
) -> None:
    from catena.experiments.config_audit import audit_configs

    payload = audit_configs(root)
    console.print_json(json.dumps(payload, ensure_ascii=False))
    if not payload["passed"]:
        raise typer.Exit(code=1)


@app.command("model-smoke")
def model_smoke(
    model: str = typer.Option(..., help="Model YAML configuration"),
    device: str = typer.Option("cuda"),
) -> None:
    from catena.models.factory import load_model

    adapter = load_model(model, device=device)
    state = adapter.prefill_text("Question: The current status is ACTIVE.\nAnswer:", None)
    scores = adapter.score_candidates(state, "Return the status.", ["ACTIVE", "INACTIVE"])
    result = {
        "model_config": model,
        "prediction": scores.prediction_index,
        "scores": scores.log_likelihoods,
        "state_bytes": adapter.state_bytes(state),
    }
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("runtime-gate")
def runtime_gate(
    config: str = typer.Option(..., help="E01 runtime YAML configuration"),
    model_index: int = typer.Option(..., min=0),
    device: str = typer.Option("cuda"),
) -> None:
    from catena.experiments.runtime_gates import run_runtime_gates

    payload = run_runtime_gates(config, model_index=model_index, device=device)
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("eval-inference")
def eval_inference(
    config: str = typer.Option(...),
    device: str = typer.Option("cuda"),
    max_episodes: Optional[int] = typer.Option(None),
    shard_index: int = typer.Option(0, min=0),
    num_shards: int = typer.Option(1, min=1),
) -> None:
    from catena.experiments.inference_eval import run

    summary = run(
        config,
        device=device,
        max_episodes=max_episodes,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    console.print_json(json.dumps(summary, ensure_ascii=False))


@app.command("teacher-cache")
def teacher_cache(
    config: str = typer.Option(...),
    split: str = typer.Option("train"),
    device: str = typer.Option("cuda"),
    shard_index: int = typer.Option(0, min=0),
    num_shards: int = typer.Option(1, min=1),
) -> None:
    from catena.training.teacher_cache import build_teacher_cache

    path = build_teacher_cache(
        config,
        device=device,
        split=split,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    console.print(f"Teacher cache written to [bold]{path}[/bold]")


@app.command("teacher-merge")
def teacher_merge(
    output_dir: str = typer.Option(...),
    split: str = typer.Option("train"),
    num_shards: int = typer.Option(4, min=1),
    remove_shards: bool = typer.Option(False),
) -> None:
    from catena.training.teacher_cache import merge_teacher_shards

    path = merge_teacher_shards(
        output_dir,
        split=split,
        num_shards=num_shards,
        remove_shards=remove_shards,
    )
    console.print(f"Merged teacher cache written to [bold]{path}[/bold]")


@app.command("train-h3")
def train_h3(
    config: str = typer.Option(...),
    seed: int = typer.Option(...),
    device: str = typer.Option("cuda"),
    max_steps: Optional[int] = typer.Option(None),
) -> None:
    from catena.training.h3_trainer import train_h3 as train

    path = train(config, seed=seed, device=device, max_steps_override=max_steps)
    console.print(f"Checkpoint written to [bold]{path}[/bold]")


@app.command("eval-h3")
def eval_h3(
    model: str = typer.Option(..., help="Model YAML configuration"),
    checkpoint: str = typer.Option(...),
    data: str = typer.Option(..., help="Episode JSONL file"),
    output: str = typer.Option(...),
    device: str = typer.Option("cuda"),
    max_episodes: Optional[int] = typer.Option(None),
    shard_index: int = typer.Option(0, min=0),
    num_shards: int = typer.Option(1, min=1),
) -> None:
    from catena.experiments.h3_eval import evaluate_h3

    summary = evaluate_h3(
        model_config=model,
        checkpoint_path=checkpoint,
        data_path=data,
        output_dir=output,
        device=device,
        max_episodes=max_episodes,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    console.print_json(json.dumps(summary, ensure_ascii=False))


@app.command("train-h4")
def train_h4_cmd(
    config: str = typer.Option(...),
    seed: int = typer.Option(...),
    device: str = typer.Option("cuda"),
    max_steps: Optional[int] = typer.Option(None),
    init_checkpoint: Optional[str] = typer.Option(
        None, help="Validated H3 checkpoint used to initialize the transaction encoder"
    ),
) -> None:
    from catena.training.h4_trainer import train_h4

    path = train_h4(
        config,
        seed=seed,
        device=device,
        max_steps_override=max_steps,
        init_checkpoint_override=init_checkpoint,
    )
    console.print(f"H4 checkpoint written to [bold]{path}[/bold]")


@app.command("eval-h4")
def eval_h4_cmd(
    model: str = typer.Option(...),
    checkpoint: str = typer.Option(...),
    data: str = typer.Option(...),
    output: str = typer.Option(...),
    device: str = typer.Option("cuda"),
    max_episodes: Optional[int] = typer.Option(None),
) -> None:
    from catena.experiments.h4_eval import evaluate_h4

    summary = evaluate_h4(
        model_config=model,
        checkpoint_path=checkpoint,
        data_path=data,
        output_dir=output,
        device=device,
        max_episodes=max_episodes,
    )
    console.print_json(json.dumps(summary, ensure_ascii=False))


@app.command("profile-system")
def profile_system(
    config: str = typer.Option(...),
    model_index: int = typer.Option(..., min=0),
    device: str = typer.Option("cuda"),
) -> None:
    from catena.experiments.profile import run_profile

    payload = run_profile(config, model_index=model_index, device=device)
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("predictions-merge")
def predictions_merge(
    input_root: str = typer.Option(...),
    filename: str = typer.Option("predictions.jsonl"),
    output_dir: Optional[str] = typer.Option(None),
) -> None:
    from catena.eval.merge import merge_prediction_shards

    path = merge_prediction_shards(
        input_root,
        filename=filename,
        output_dir=output_dir,
    )
    console.print(f"Merged predictions written to [bold]{path}[/bold]")


@app.command("eval-toolcalls")
def eval_toolcalls(
    config: str = typer.Option(...),
    run_index: int = typer.Option(..., min=0),
    device: str = typer.Option("cuda"),
    max_episodes: Optional[int] = typer.Option(None),
) -> None:
    from catena.experiments.toolcall_eval import run_toolcall_eval

    payload = run_toolcall_eval(
        config,
        run_index=run_index,
        device=device,
        max_episodes=max_episodes,
    )
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("experiment-list")
def experiment_list() -> None:
    """Display the chronological experiment sequence."""
    rows = [
        ("E00", "Environment audit", "Driver/CUDA/PyTorch/topology/storage"),
        ("E01", "Runtime parity", "RWKV state and Transformer cache correctness"),
        ("E02", "Data and leakage audit", "Schema, closure, Tx-only checks"),
        ("E03", "H1 stale failure", "Stale state versus exact refresh"),
        ("E04", "H2 representation", "Plain, typed, closure, reset, retrieval"),
        ("E05", "Teacher cache", "Exact candidate distributions"),
        ("E06", "H3 pilot and slot sweep", "K=4/8/16 plus generic control"),
        ("E07", "H3 main and ablations", "Three seeds and coherence-cost Pareto"),
        ("E08", "Transformer boundary", "Append, capsule, suffix/full reprefill, soft patch"),
        ("E09", "H4 composition", "Long-chain drift and unseen combinations"),
        ("E10", "System profile", "Latency, bytes, TTFA, decode"),
        ("E11", "Naturalized/tool calls", "Schema validity and simulator success"),
        ("E12", "Clean rerun", "Claim gates, figures, reproducibility"),
    ]
    table = Table("ID", "Experiment", "Primary output")
    for row in rows:
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
