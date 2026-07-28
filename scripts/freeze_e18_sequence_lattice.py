#!/usr/bin/env python3
"""Freeze one provenance-valid E18b MAIN aggregate without changing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import write_e18_result_summaries as summaries

FREEZE_FILENAME = "E18_SEQUENCE_CONTROL_LATTICE_FREEZE_V1.json"
SUMMARY_FILENAME = summaries.SUMMARY_FILENAME
FROZEN_ARTIFACT_FILES = (
    "report.json",
    "run_manifest.json",
    "sequence_control_lattice_paired_metrics.jsonl",
    "sequence_control_lattice_active_path_metrics.jsonl",
    "source_run_provenance.jsonl",
    SUMMARY_FILENAME,
)
FROZEN_HASH_KEYS = frozenset(
    {
        *FROZEN_ARTIFACT_FILES,
        "protocol_lock",
        "source_config",
        "aggregate_config",
    }
)
AUDITED_ALLOWED_CLAIM_IF_SUPPORTED = (
    "In a controlled associative-memory sequence probe with oracle "
    "erase/write addresses, oracle candidates, explicit oracle demand "
    "descriptors, and a model-visible verified-event bit, each adjacent "
    "controller expansion achieved the preregistered mean affected-MSE gain "
    "on its matched demand over the registered update-by-gap grid. "
    "Simpler-demand and retention degradation remained within the relative "
    "cell-mean adjacent non-inferiority margins. At the registered "
    "8-update/2,048-distractor stress cell, matched gain was positive in all "
    "five paired training seeds."
)
AUDITED_FORBIDDEN_CLAIM = (
    "Positive gain in every grid cell, uniform persistence, a registered "
    "stress SESOI, absolute or accurate preservation, exclusive benefit only "
    "on the matched demand, minimal-controller sufficiency, semantic demand "
    "or relevance inference, learned address/candidate recovery, arbitrary "
    "event interleaving, natural-language, recurrent-LM, pretrained-model, "
    "agent/planning, or official-backend transfer."
)


def _audited_claim_gate(supported: bool) -> dict[str, Any]:
    return {
        "claim_eligible": bool(supported),
        "status": "SUPPORTED" if supported else "NOT_OPENED",
        "primary_estimand": (
            "matched adjacent registered-grid mean affected-MSE gain"
        ),
        "stress_interpretation": (
            "positive direction in 5/5 paired seeds at updates=8, gap=2048"
        ),
        "stress_separate_sesoi_registered": False,
        "guardrail_interpretation": (
            "relative adjacent non-inferiority of cell-mean simpler-demand "
            "and retention MSE; not an absolute-accuracy gate"
        ),
        "input_boundary": {
            "oracle_erase_write_addresses": True,
            "oracle_candidates": True,
            "explicit_oracle_demand_descriptors": True,
            "model_visible_verified_event_bit": True,
        },
        "allowed_claim_if_supported": AUDITED_ALLOWED_CLAIM_IF_SUPPORTED,
        "forbidden_claim": AUDITED_FORBIDDEN_CLAIM,
    }


def _claim_boundary(supported: bool) -> dict[str, bool]:
    return {
        "controlled_sequence_lattice_claim_eligible": bool(supported),
        "registered_grid_mean_claim_eligible": bool(supported),
        "stress_five_of_five_direction_claim_eligible": bool(supported),
        "every_cell_or_uniform_persistence_claim_eligible": False,
        "stress_sesoi_claim_eligible": False,
        "absolute_accurate_preservation_claim_eligible": False,
        "minimal_controller_sufficiency_claim_eligible": False,
        "semantic_demand_or_relevance_inference_claim_eligible": False,
        "learned_localization_or_candidate_claim_eligible": False,
        "natural_language_claim_eligible": False,
        "pretrained_model_claim_eligible": False,
        "official_backend_claim_eligible": False,
        "agent_claim_eligible": False,
        "oracle_address": True,
        "oracle_candidate": True,
        "explicit_oracle_demand_descriptors": True,
        "model_visible_verified_event_bit": True,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def build_freeze_payload(
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    artifacts = Path(artifact_root).resolve()
    plan = summaries.build_summary_plan(
        repo_root=root,
        artifact_root=artifacts,
        scope="aggregate",
        include_live=True,
    )
    aggregate_actions = [
        action for action in plan.actions if action.kind == "AGGREGATE"
    ]
    if not plan.aggregate_available or len(aggregate_actions) != 1:
        raise RuntimeError(
            "one provenance-valid E18b MAIN aggregate is required"
        )
    action = aggregate_actions[0]
    run_dir = action.run_dir.resolve()
    expected_parent = (
        artifacts / summaries.AGGREGATE_EXPERIMENT_ID
    ).resolve()
    if run_dir.parent != expected_parent:
        raise RuntimeError(f"E18b run escapes canonical namespace: {run_dir}")

    summary_path = run_dir / SUMMARY_FILENAME
    if not summary_path.is_file():
        raise RuntimeError(
            "E18b compact summary must be created and validated before freeze"
        )
    if summary_path.read_text(encoding="utf-8") != action.content:
        raise RuntimeError(f"E18b compact summary content mismatch: {summary_path}")

    report_path = run_dir / "report.json"
    manifest_path = run_dir / "run_manifest.json"
    report = _read_json(report_path)
    manifest = _read_json(manifest_path)
    if (
        report.get("status") != "PASS"
        or report.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or report.get("scientific_evidence") is not False
        or manifest.get("run_mode") != "MAIN"
        or not isinstance(report.get("claim_gate", {}).get("supported"), bool)
    ):
        raise RuntimeError("E18 aggregate claim/evidence boundary is invalid")
    supported = report["claim_gate"]["supported"]
    status = "SUPPORTED" if supported else "NOT_OPENED"

    hashes = {
        name: _sha256(run_dir / name) for name in FROZEN_ARTIFACT_FILES
    }
    hashes.update(
        {
            "protocol_lock": _sha256(
                root / "docs/E18_SEQUENCE_CONTROL_LATTICE_LOCK.json"
            ),
            "source_config": _sha256(
                root / "configs/e18a_sequence_control_lattice.yaml"
            ),
            "aggregate_config": _sha256(
                root
                / "configs/e18b_sequence_control_lattice_aggregate.yaml"
            ),
        }
    )
    if hashes["protocol_lock"] != plan.protocol_lock_sha256:
        raise RuntimeError("E18 protocol lock hash changed during freeze")

    timestamp = frozen_at_utc or datetime.now(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": 1,
        "frozen_at_utc": timestamp,
        "experiment_family": "E18",
        "aggregate_experiment_id": summaries.AGGREGATE_EXPERIMENT_ID,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "execution_status": str(report["status"]),
        "claim_status": status,
        "evidence_tier": str(report["evidence_tier"]),
        "scientific_evidence": bool(report["scientific_evidence"]),
        "summary": report["summary"],
        "contrasts": report["contrasts"],
        # Preserve the preregistered report verbatim.  Its prose is historical
        # source evidence, not the post-audit writing boundary.
        "registered_report_claim_gate": report["claim_gate"],
        "audited_claim_gate": _audited_claim_gate(supported),
        "source_manifest": {
            "run_mode": manifest["run_mode"],
            "source_fingerprint": manifest["source_fingerprint"],
            "source_fingerprint_phase": manifest[
                "source_fingerprint_phase"
            ],
        },
        "hashes": hashes,
        "claim_boundary": _claim_boundary(supported),
        "immutable": True,
    }


def validate_freeze(
    payload: dict[str, Any],
    *,
    repo_root: str | Path,
    artifact_root: str | Path,
) -> None:
    required_identity = {
        "schema_version": 1,
        "experiment_family": "E18",
        "aggregate_experiment_id": summaries.AGGREGATE_EXPERIMENT_ID,
        "immutable": True,
    }
    for key, expected in required_identity.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"invalid E18 freeze field: {key}")
    if not isinstance(payload.get("frozen_at_utc"), str):
        raise RuntimeError("invalid E18 freeze timestamp")
    expected = build_freeze_payload(
        repo_root=repo_root,
        artifact_root=artifact_root,
        frozen_at_utc=str(payload["frozen_at_utc"]),
    )
    if payload != expected:
        raise RuntimeError("E18 freeze does not reproduce from frozen artifacts")
    artifact_root_path = Path(artifact_root).resolve()
    run_id = payload.get("run_id")
    if (
        not isinstance(run_id, str)
        or summaries.RUN_ID_PATTERN.fullmatch(run_id) is None
        or Path(str(payload.get("run_dir"))).resolve()
        != (
            artifact_root_path
            / summaries.AGGREGATE_EXPERIMENT_ID
            / run_id
        ).resolve()
    ):
        raise RuntimeError("invalid E18 freeze run identity")
    timestamp = str(payload["frozen_at_utc"])
    try:
        parsed_timestamp = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise RuntimeError("invalid E18 freeze timestamp") from error
    if parsed_timestamp.tzinfo is None:
        raise RuntimeError("E18 freeze timestamp must include UTC offset")

    registered_claim_gate = payload.get("registered_report_claim_gate")
    if not isinstance(registered_claim_gate, dict) or not isinstance(
        registered_claim_gate.get("supported"),
        bool,
    ):
        raise RuntimeError("invalid E18 registered report claim gate")
    supported = registered_claim_gate["supported"]
    expected_claim_status = "SUPPORTED" if supported else "NOT_OPENED"
    if (
        payload.get("execution_status") != "PASS"
        or payload.get("claim_status") != expected_claim_status
        or payload.get("evidence_tier") != "CONTROLLED_REFERENCE"
        or payload.get("scientific_evidence") is not False
        or payload.get("audited_claim_gate")
        != _audited_claim_gate(supported)
        or payload.get("claim_boundary") != _claim_boundary(supported)
    ):
        raise RuntimeError("invalid E18 freeze claim/evidence boundary")
    hashes = payload.get("hashes")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != FROZEN_HASH_KEYS
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise RuntimeError("invalid E18 freeze hash inventory")


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    content += "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/home/minjun_dev/CATENA")
    parser.add_argument(
        "--artifact-root",
        default=os.getenv(
            "CATENA_ARTIFACT_ROOT",
            "/data/minjun_dev/CATENA/artifacts",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    output_path = artifact_root / FREEZE_FILENAME
    try:
        if args.validate_existing:
            validate_freeze(
                _read_json(output_path),
                repo_root=args.repo_root,
                artifact_root=artifact_root,
            )
            print(
                f"[E18 FREEZE] VALID {output_path} sha256={_sha256(output_path)}"
            )
            return 0
        payload = build_freeze_payload(
            repo_root=args.repo_root,
            artifact_root=artifact_root,
        )
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
            return 0
        _write_exclusive(output_path, payload)
        print(
            f"[E18 FREEZE] CREATED {output_path} sha256={_sha256(output_path)}"
        )
        return 0
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
