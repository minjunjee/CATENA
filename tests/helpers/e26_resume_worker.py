from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from catena.lm.audit_contract import E26_AUDIT_LOCKED_HASH_KEYS
from catena.lm.checkpointing import (
    TrainingProgress,
    load_training_checkpoint,
    runtime_state_to_payload,
    save_training_checkpoint,
)
from catena.lm.config import ModelConfig
from catena.lm.general_corpus import TokenMemmap, write_synthetic_token_memmap
from catena.lm.hashing import state_dict_digest, tensor_tree_digest
from catena.lm.model import CatenaLM
from catena.lm.paired_stream import PairedTrainingCursor, PairedTransactionCursor
from catena.lm.tokenizer import ByteTokenizer
from catena.lm.trainer import make_optimizer, optimizer_step_microbatches

LOCKED_HASHES = {
    key: f"{index:064x}" for index, key in enumerate(sorted(E26_AUDIT_LOCKED_HASH_KEYS), start=1)
}


def build(root: Path):
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    config = ModelConfig.tiny_reference()
    model = CatenaLM(config)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - 0.01 * step),
    )
    manifest = write_synthetic_token_memmap(
        root / "corpus",
        vocab_size=config.vocab_size,
        token_count=4_096,
        seed=94,
    )
    corpus = TokenMemmap(manifest)
    tokenizer = ByteTokenizer()
    tokenizer_hash = tokenizer.manifest().manifest_hash
    cursor = PairedTrainingCursor(
        corpus.paired_cursor(seed=95, sequence_length=16),
        PairedTransactionCursor(
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            seed=96,
            sequence_length=16,
            pad_token_id=tokenizer.pad_id,
        ),
    )
    return model, optimizer, scheduler, corpus, tokenizer, tokenizer_hash, cursor


def run_steps(model, optimizer, scheduler, cursor, steps: int):
    losses: list[float] = []
    sources: list[str] = []
    for _ in range(steps):
        rows, _ = cursor.take(1)
        sources.append(rows[0].source_type)
        batch = torch.tensor(np.stack([rows[0].token_ids]), dtype=torch.long)
        result = optimizer_step_microbatches(
            model,
            [batch],
            optimizer=optimizer,
            scheduler=scheduler,
        )
        losses.append(result.loss)
    return losses, sources


def final_record(model, optimizer, scheduler, cursor, loss: float, sources: list[str]):
    probe = torch.arange(16, dtype=torch.long)[None, :] % model.config.vocab_size
    model.eval()
    with torch.no_grad():
        output = model(probe)
    return {
        "model": state_dict_digest(model),
        "optimizer": tensor_tree_digest(optimizer.state_dict()),
        "scheduler": tensor_tree_digest(scheduler.state_dict()),
        "cursor": cursor.snapshot()["snapshot_sha256"],
        "loss": loss,
        "logits": tensor_tree_digest(output.logits),
        "runtime": tensor_tree_digest(runtime_state_to_payload(output.runtime_state)),
        "python_rng": random.random(),
        "numpy_rng": float(np.random.random()),
        "torch_rng": float(torch.rand(()).item()),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "save", "resume"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    model, optimizer, scheduler, corpus, tokenizer, tokenizer_hash, cursor = build(args.root)

    if args.mode == "baseline":
        losses, sources = run_steps(model, optimizer, scheduler, cursor, args.n + args.m)
        record = final_record(model, optimizer, scheduler, cursor, losses[-1], sources)
    elif args.mode == "save":
        losses, sources = run_steps(model, optimizer, scheduler, cursor, args.n)
        progress = TrainingProgress(
            optimizer_step=args.n,
            tokens_seen=args.n * cursor.sequence_length,
            general_sequences_seen=cursor.general.sequence_index,
            transaction_sequences_seen=cursor.transaction.sequence_index,
            document_index=cursor.general.sequence_index,
            episode_index=cursor.transaction.sequence_index,
            cursor_snapshot=cursor.snapshot(),
            last_source_type=sources[-1],
        )
        receipt = save_training_checkpoint(
            args.root / "resume.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            locked_hashes=LOCKED_HASHES,
        )
        record = {
            "checkpoint": receipt.as_dict(),
            "last_source": sources[-1],
            "loss": losses[-1],
        }
    else:
        if args.receipt is None:
            raise ValueError("--receipt is required for resume")
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))["checkpoint"]
        loaded = load_training_checkpoint(
            receipt["path"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_locked_hashes=LOCKED_HASHES,
            expected_file_sha256=receipt["sha256"],
        )
        cursor = PairedTrainingCursor.from_snapshot(
            corpus,
            tokenizer,
            tokenizer_hash=tokenizer_hash,
            snapshot=loaded.progress.cursor_snapshot,
        )
        losses, sources = run_steps(model, optimizer, scheduler, cursor, args.m)
        record = final_record(model, optimizer, scheduler, cursor, losses[-1], sources)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
