from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from catena.core.config import load_config
from catena.core.provenance_v61 import (
    ProvenanceValidationError,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)
from catena.eval.semantic_audit_v61 import (
    SemanticAuditThresholds,
    evaluate_semantic_human_audit,
)
from experiments.common import build_parser
from experiments.e05_common_v61 import (
    PINNED_PROTOCOL_LOCK_SHA256,
    PINNED_PROTOCOL_SHA256,
    validate_completed_e05a_run,
    validate_frozen_e05_protocol,
)
from experiments.v61_common import (
    finalize_v61_run,
    initialize_v61_run,
    validate_legacy_e00,
)

EXPERIMENT_ID = "e05a_semantic_audit_adjudication"
DEFAULT_CONFIG = "configs/e05a_semantic_audit_adjudication.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]

_CONFIG_CANONICAL_SHA256 = (
    "c564a2058d7647ebdf2eb38b022622fc0c6b47f4754dd3cb57c951fa16cb98b6"
)
_CONFIG_FILE_SHA256 = (
    "c43e5cd6914b22006bd5ca9195504d4cec7d6c6478c9eb8efba111f04f82eb7c"
)
_IMPLEMENTATION_LOCK_SHA256 = (
    "19392aaced5859bf1a7d69de936e49e198e71bc7867633ef019cd1260aa4f4ab"
)


def _validate_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).resolve(strict=True)
    expected_path = (REPO_ROOT / DEFAULT_CONFIG).resolve(strict=True)
    if path != expected_path:
        raise ProvenanceValidationError(
            "Human audit runner requires the frozen default config path."
        )
    lock = REPO_ROOT / "docs/E05A_HUMAN_AUDIT_ADJUDICATION_LOCK_KO.md"
    if sha256_file(lock) != _IMPLEMENTATION_LOCK_SHA256:
        raise ProvenanceValidationError("Human audit implementation lock changed.")
    if sha256_file(path) != _CONFIG_FILE_SHA256:
        raise ProvenanceValidationError("Human audit config byte hash changed.")
    config = load_config(path)
    if sha256_canonical_json(config) != _CONFIG_CANONICAL_SHA256:
        raise ProvenanceValidationError("Human audit canonical config hash changed.")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ProvenanceValidationError("Human audit experiment identity changed.")
    return config


def _audit_items_from_e05a(run: Any) -> Path:
    artifacts = run.report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ProvenanceValidationError("E05a report lacks an artifacts registry.")
    descriptor = artifacts.get("naturalization_audit_items")
    if not isinstance(descriptor, dict):
        raise ProvenanceValidationError(
            "E05a report lacks naturalization_audit_items."
        )
    filename = descriptor.get("filename")
    expected_hash = descriptor.get("sha256")
    expected_rows = descriptor.get("rows")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ProvenanceValidationError("Unsafe E05a audit-item filename.")
    if not isinstance(expected_hash, str):
        raise ProvenanceValidationError("E05a audit-item hash is missing.")
    if expected_rows != 300:
        raise ProvenanceValidationError("E05a audit-item row count is not 300.")
    path = run.run_dir / filename
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
        raise ProvenanceValidationError("E05a audit-item artifact hash mismatch.")
    return path


def _regular_external_file(
    path_string: str,
    label: str,
    *,
    forbidden_roots: tuple[Path, ...] = (),
) -> Path:
    path = Path(path_string).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a direct regular file.")
    for root in forbidden_roots:
        resolved_root = root.expanduser().resolve(strict=True)
        if path.is_relative_to(resolved_root):
            raise ValueError(
                f"{label} must be an external copy outside immutable artifact "
                f"root {resolved_root}."
            )
    return path


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    parser.add_argument("--e05a-run-dir", required=True)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--adjudication", required=True)
    args = parser.parse_args()
    if args.dry_run:
        raise ValueError("Human adjudication does not support a dry-run alias.")

    validate_frozen_e05_protocol()
    config = _validate_config(args.config)
    e05a = validate_completed_e05a_run(args.e05a_run_dir, require_go=True)
    audit_items = _audit_items_from_e05a(e05a)
    immutable_roots = (
        Path(args.artifact_root),
        e05a.run_dir,
    )
    reviewer_a = _regular_external_file(
        args.reviewer_a,
        "reviewer A",
        forbidden_roots=immutable_roots,
    )
    reviewer_b = _regular_external_file(
        args.reviewer_b,
        "reviewer B",
        forbidden_roots=immutable_roots,
    )
    adjudication = _regular_external_file(
        args.adjudication,
        "adjudication",
        forbidden_roots=immutable_roots,
    )
    if len({reviewer_a, reviewer_b, adjudication}) != 3:
        raise ValueError("Reviewer and adjudication inputs must be three distinct files.")

    e00_dependency = validate_legacy_e00(args.artifact_root, require_full=True)
    dependencies = [
        e00_dependency,
        {
            **e05a.dependency_record(),
            "evidence_role": "e05a_go_and_locked_audit_item_source",
            "e05a_design_status": "GO",
            "protocol_sha256": PINNED_PROTOCOL_SHA256,
            "protocol_lock_sha256": PINNED_PROTOCOL_LOCK_SHA256,
        },
    ]
    _, run_dir, _, context = initialize_v61_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request="cpu",
        dry_run=False,
        dependencies=dependencies,
    )

    copied = {
        "audit_items.csv": audit_items,
        "reviewer_a.csv": reviewer_a,
        "reviewer_b.csv": reviewer_b,
        "adjudication.csv": adjudication,
    }
    for filename, source in copied.items():
        shutil.copyfile(source, run_dir / filename)

    threshold_config = config["thresholds"]
    audit_report = evaluate_semantic_human_audit(
        audit_items_path=run_dir / "audit_items.csv",
        reviewer_a_path=run_dir / "reviewer_a.csv",
        reviewer_b_path=run_dir / "reviewer_b.csv",
        adjudication_path=run_dir / "adjudication.csv",
        thresholds=SemanticAuditThresholds(
            total_items=int(threshold_config["total_items"]),
            minimum_meaning_preservation=float(
                threshold_config["minimum_meaning_preservation"]
            ),
            maximum_answer_leakage=float(
                threshold_config["maximum_answer_leakage"]
            ),
            minimum_raw_agreement_each_label=float(
                threshold_config["minimum_raw_agreement_each_label"]
            ),
        ),
    )
    artifact_registry = {
        filename: {
            "filename": filename,
            "sha256": sha256_file(run_dir / filename),
        }
        for filename in copied
    }
    write_json_strict(run_dir / "audit_artifact_registry.json", artifact_registry)
    report = {
        "status": "PASS",
        "execution_status": "PASS",
        "human_audit_status": "PASSED" if audit_report.passed else "FAILED",
        "human_audit": audit_report.to_dict(),
        "artifacts": {
            **artifact_registry,
            "audit_artifact_registry.json": {
                "filename": "audit_artifact_registry.json",
                "sha256": sha256_file(run_dir / "audit_artifact_registry.json"),
            },
        },
        "protocol_lock": {
            "protocol_sha256": PINNED_PROTOCOL_SHA256,
            "protocol_lock_sha256": PINNED_PROTOCOL_LOCK_SHA256,
            "implementation_lock_sha256": _IMPLEMENTATION_LOCK_SHA256,
        },
        "claim_gate": {
            "opens_h5_claim": False,
            "is_e05b_training_dependency": audit_report.passed,
        },
        "evidence_scope": {
            "evidence_tier": "CONTROLLED_REFERENCE",
            "scientific_evidence": False,
        },
    }
    finalize_v61_run(
        context=context,
        report=report,
        main_eligible=True,
        full_eligible=True,
    )
    print(f"[{EXPERIMENT_ID}] PASS: {run_dir}")


if __name__ == "__main__":
    main()
