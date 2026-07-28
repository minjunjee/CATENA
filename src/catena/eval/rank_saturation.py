from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_checkpoint_index_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_run_dir: Path,
) -> str:
    """Hash the registered checkpoint identity and digest set canonically."""
    source_run_dir = source_run_dir.resolve()
    normalized: list[dict[str, int | str]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["seed"]),
            int(item["intrinsic_rank"]),
            int(item["learned_rank"]),
        ),
    ):
        checkpoint = Path(str(row["checkpoint"])).resolve()
        try:
            relative = checkpoint.relative_to(source_run_dir)
        except ValueError as exc:
            raise ValueError(
                f"checkpoint escapes the frozen source run: {checkpoint}"
            ) from exc
        normalized.append(
            {
                "seed": int(row["seed"]),
                "intrinsic_rank": int(row["intrinsic_rank"]),
                "learned_rank": int(row["learned_rank"]),
                "checkpoint": relative.as_posix(),
                "sha256": str(row["checkpoint_sha256"]),
            }
        )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_pre_saturation_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    recovery_threshold: float,
) -> list[dict[str, int | float | bool | str]]:
    """Classify adjacent learned-rank pairs using the lower-rank recovery only.

    A pair is eligible for the prospective monotonicity gate exactly when its
    lower-rank controller has not yet reached the registered exact-target
    recovery threshold. Pairs whose lower rank already qualifies are retained
    as transparent ``SATURATED_EXCLUDED`` diagnostics.
    """
    threshold = float(recovery_threshold)
    if not math.isfinite(threshold):
        raise ValueError("recovery_threshold must be finite")
    if len(rows) < 2:
        raise ValueError("at least two learned-rank rows are required")

    ordered = sorted(rows, key=lambda item: int(item["learned_rank"]))
    ranks = [int(item["learned_rank"]) for item in ordered]
    if any(rank <= 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("learned ranks must be unique positive integers")

    pairs: list[dict[str, int | float | bool | str]] = []
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        lower_error = float(lower["test_error"])
        upper_error = float(upper["test_error"])
        lower_recovery = float(lower["exact_target_recovery"])
        upper_recovery = float(upper["exact_target_recovery"])
        if not all(
            math.isfinite(value)
            for value in (
                lower_error,
                upper_error,
                lower_recovery,
                upper_recovery,
            )
        ):
            raise ValueError("rank-pair metrics must be finite")
        if lower_error < 0.0 or upper_error < 0.0:
            raise ValueError("rank-pair errors must be non-negative")

        eligible = lower_recovery < threshold
        non_increasing = upper_error <= lower_error
        pairs.append(
            {
                "lower_rank": int(lower["learned_rank"]),
                "upper_rank": int(upper["learned_rank"]),
                "lower_error": lower_error,
                "upper_error": upper_error,
                "error_decrease": lower_error - upper_error,
                "lower_exact_target_recovery": lower_recovery,
                "upper_exact_target_recovery": upper_recovery,
                "eligible_pre_saturation": eligible,
                "non_increasing": non_increasing,
                "gate_passed": (not eligible) or non_increasing,
                "pair_disposition": (
                    "ELIGIBLE_PRE_SATURATION"
                    if eligible
                    else "SATURATED_EXCLUDED"
                ),
            }
        )
    return pairs


def eligible_pre_saturation_monotonic_fraction(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[float, int, int]:
    """Return passing fraction, passing count and eligible pair count."""
    eligible = [item for item in pairs if bool(item["eligible_pre_saturation"])]
    if not eligible:
        raise ValueError("pre-saturation monotonicity is unidentifiable: no eligible pairs")
    passed = sum(bool(item["non_increasing"]) for item in eligible)
    return passed / len(eligible), passed, len(eligible)
