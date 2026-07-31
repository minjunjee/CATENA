from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from catena.lm.audit_contract import E26_AUDIT_LOCKED_HASH_KEYS
from catena.lm.checkpointing import (
    TrainingCheckpointError,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from catena.lm.config import ModelConfig
from catena.lm.hashing import state_dict_digest, tensor_tree_digest
from catena.lm.model import CatenaLM
from catena.lm.trainer import make_optimizer


def test_checkpoint_round_trip_restores_all_rng_and_training_state(
    tmp_path: Path,
) -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    model = CatenaLM(ModelConfig.tiny_reference())
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    batch = torch.randint(0, model.config.vocab_size, (1, 12))
    loss = model(batch).logits.square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    progress = TrainingProgress(
        optimizer_step=1,
        tokens_seen=12,
        general_sequences_seen=1,
        transaction_sequences_seen=0,
        document_index=1,
        episode_index=0,
        cursor_snapshot={"snapshot_sha256": "e" * 64},
        last_source_type="general",
    )
    locked = {
        key: f"{index:064x}"
        for index, key in enumerate(sorted(E26_AUDIT_LOCKED_HASH_KEYS), start=1)
    }
    expected_model = state_dict_digest(model)
    expected_optimizer = tensor_tree_digest(optimizer.state_dict())
    expected_scheduler = tensor_tree_digest(scheduler.state_dict())
    training_graph_identity = {
        "last_graph_code_sha256": "a" * 64,
        "last_graph_node_count": 8,
        "positive_compiled_execution": True,
        "fallback_count": 0,
        "graph_break_count": 0,
        "passed": True,
    }
    backend_manifest = {
        "backend_id": "reference_python_test_only",
        "compiler": "none",
        "source_tree_sha256": locked["source_tree_sha256"],
        "training_graph_identity": training_graph_identity,
    }
    receipt = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        locked_hashes=locked,
        backend_manifest=backend_manifest,
    )
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(()).item()))
    with torch.no_grad():
        next(model.parameters()).add_(3)
    random.seed(100)
    np.random.seed(101)
    torch.manual_seed(102)

    loaded = load_training_checkpoint(
        receipt.path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_locked_hashes=locked,
        expected_file_sha256=receipt.sha256,
        expected_backend_manifest=backend_manifest,
    )
    observed_rng = (random.random(), float(np.random.random()), float(torch.rand(()).item()))
    assert observed_rng == expected_rng
    assert loaded.progress == progress
    assert state_dict_digest(model) == expected_model
    assert tensor_tree_digest(optimizer.state_dict()) == expected_optimizer
    assert tensor_tree_digest(scheduler.state_dict()) == expected_scheduler
    assert loaded.backend_manifest == backend_manifest

    with pytest.raises(TrainingCheckpointError, match="hashes changed"):
        load_training_checkpoint(
            receipt.path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_locked_hashes={**locked, "data_lock_sha256": "f" * 64},
            expected_file_sha256=receipt.sha256,
        )

    with pytest.raises(TrainingCheckpointError, match="backend manifest changed"):
        load_training_checkpoint(
            receipt.path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_locked_hashes=locked,
            expected_file_sha256=receipt.sha256,
            expected_backend_manifest={
                **backend_manifest,
                "compiler": "unexpected",
            },
        )

    with pytest.raises(TrainingCheckpointError, match="backend manifest changed"):
        load_training_checkpoint(
            receipt.path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_locked_hashes=locked,
            expected_file_sha256=receipt.sha256,
            expected_backend_manifest={
                **backend_manifest,
                "training_graph_identity": {
                    **training_graph_identity,
                    "last_graph_code_sha256": "b" * 64,
                },
            },
        )


@pytest.mark.parametrize(
    ("boundary", "last_before", "first_after"),
    [
        (4, "general", "transaction"),
        (5, "transaction", "general"),
    ],
)
def test_continuous_and_new_process_resume_are_identical(
    tmp_path: Path,
    boundary: int,
    last_before: str,
    first_after: str,
) -> None:
    worker = Path(__file__).parent / "helpers" / "e26_resume_worker.py"
    case_root = tmp_path / f"boundary_{boundary}"
    case_root.mkdir()
    environment = dict(os.environ)
    source_root = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    )

    def invoke(mode: str, output: Path, *, receipt: Path | None = None) -> None:
        command = [
            sys.executable,
            str(worker),
            "--mode",
            mode,
            "--root",
            str(case_root),
            "--n",
            str(boundary),
            "--m",
            "2",
            "--output",
            str(output),
        ]
        if receipt is not None:
            command.extend(["--receipt", str(receipt)])
        subprocess.run(command, check=True, env=environment, capture_output=True, text=True)

    baseline_path = case_root / "baseline.json"
    save_path = case_root / "save.json"
    resumed_path = case_root / "resumed.json"
    invoke("baseline", baseline_path)
    invoke("save", save_path)
    invoke("resume", resumed_path, receipt=save_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    saved = json.loads(save_path.read_text(encoding="utf-8"))
    resumed = json.loads(resumed_path.read_text(encoding="utf-8"))

    assert saved["last_source"] == last_before
    assert resumed["sources"][0] == first_after
    baseline_core = {key: value for key, value in baseline.items() if key != "sources"}
    resumed_core = {key: value for key, value in resumed.items() if key != "sources"}
    assert resumed_core == baseline_core
