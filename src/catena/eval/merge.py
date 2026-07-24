from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from catena.eval.metrics import PredictionRecord, stratified_summary


def merge_prediction_shards(
    input_root: str | Path,
    *,
    filename: str = "predictions.jsonl",
    output_dir: str | Path | None = None,
) -> Path:
    root = Path(input_root)
    output = Path(output_dir) if output_dir else root / "merged"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.glob(f"shard_*_of_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No shard files named {filename} under {root}")
    field_names = {field.name for field in fields(PredictionRecord)}
    by_policy: dict[str, list[PredictionRecord]] = {}
    seen: set[tuple[str, str, str]] = set()
    merged_path = output / filename
    with merged_path.open("w", encoding="utf-8") as writer:
        for path in paths:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    payload: dict[str, Any] = json.loads(line)
                    key = (
                        str(payload["episode_id"]),
                        str(payload["query_id"]),
                        str(payload["policy"]),
                    )
                    if key in seen:
                        raise ValueError(f"Duplicate prediction row: {key}")
                    seen.add(key)
                    record = PredictionRecord(
                        **{name: payload.get(name) for name in field_names}
                    )
                    by_policy.setdefault(record.policy, []).append(record)
                    writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
    summary = {
        "shards": [str(path) for path in paths],
        "records": len(seen),
        "policies": {
            policy: stratified_summary(records)
            for policy, records in by_policy.items()
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return merged_path
