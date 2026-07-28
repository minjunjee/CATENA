from __future__ import annotations

import os
from pathlib import Path

from catena.core.schema import Operation
from catena.models.official_adapters import (
    KVEraserAdapter,
    OfficialBackendConfig,
    OfficialBackendNotConfigured,
)
from experiments.common import build_parser, finalize_run, initialize_run

EXPERIMENT_ID = "e07_transformer_boundary"
DEFAULT_CONFIG = "configs/e07_transformer_boundary.yaml"


def main() -> None:
    parser = build_parser(EXPERIMENT_ID, DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["mock", "official"], default="mock")
    args = parser.parse_args()
    config, run_dir, _ = initialize_run(
        experiment_id=EXPERIMENT_ID,
        config_path=args.config,
        artifact_root=args.artifact_root,
        device_request=args.device,
    )
    policies = {
        Operation.PRESERVE.value: "kv_reuse",
        Operation.ADD.value: "typed_append",
        Operation.INVALIDATE.value: "kveraser",
        Operation.SUPERSEDE.value: "kveraser_then_typed_append",
        "exact": "affected_suffix_or_full_reprefill",
    }
    fairness = {
        "localization_condition": config["fairness"]["localization_condition"],
        "report_own_oracle_agreement": True,
        "include_localization_cost_when_non_oracle": True,
        "avoid_raw_accuracy_ranking_across_backbones": True,
    }
    if args.mode == "mock":
        report = {
            "status": "MOCK_ONLY",
            "operation_matched_policies": policies,
            "fairness": fairness,
            "scientific_evidence": False,
            "next_step": (
                "Pin the official KVEraser repository and implement the adapter without "
                "changing the operation-matched protocol."
            ),
        }
    else:
        repository = Path(
            os.getenv("CATENA_KVERASER_REPO", config["official_backend"]["repository"])
        )
        revision = str(config["official_backend"]["revision"])
        try:
            adapter = KVEraserAdapter(
                OfficialBackendConfig(
                    repository=repository,
                    revision=revision,
                    backend_name="KVEraser",
                )
            )
            readiness = adapter.readiness()
            report = {
                "status": "ADAPTER_READY_EXECUTION_NOT_IMPLEMENTED",
                "operation_matched_policies": policies,
                "fairness": fairness,
                "backend": readiness,
                "scientific_evidence": False,
            }
        except OfficialBackendNotConfigured as exc:
            report = {
                "status": "BLOCKED",
                "reason": str(exc),
                "operation_matched_policies": policies,
                "fairness": fairness,
                "scientific_evidence": False,
            }
    finalize_run(
        experiment_id=EXPERIMENT_ID,
        artifact_root=args.artifact_root,
        run_dir=run_dir,
        report=report,
    )
    print(f"[{EXPERIMENT_ID}] {report['status']}: {run_dir}")


if __name__ == "__main__":
    main()
