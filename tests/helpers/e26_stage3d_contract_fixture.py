from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3C_ROOT = Path(
    "/data/minjun_dev/CATENA/artifacts/e26_stage3c_numerical_preflight/20260802T060323Z"
)
RESULT = REPO_ROOT / "docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_RESULT_KO.md"
STATUS = REPO_ROOT / "docs/E26_STAGE3C_FINAL_DATA_PREFLIGHT_STATUS.json"
RAW_FILES = (
    "d448_ctx4096.log",
    "d448_ctx4096_numerical.json",
    "d448_ctx4096_worker_spec.json",
    "d512_ctx2048.log",
    "d512_ctx2048_numerical.json",
    "d512_ctx2048_worker_spec.json",
    "d512_ctx4096.log",
    "d512_ctx4096_numerical.json",
    "d512_ctx4096_worker_spec.json",
    "failure_status.json",
    "source_lock.json",
)
REGISTERED_RAW_AGGREGATE_SHA256 = "296556071853073cfdf678a114d95e61cc5d21d46caa2ab97a111eca508417cc"


def _aggregate(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def stage3c_fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    raw_root = tmp_path / "stage3c_raw"
    raw_root.mkdir()
    rows: list[dict[str, Any]] = []
    for name in RAW_FILES:
        destination = raw_root / name
        shutil.copyfile(STAGE3C_ROOT / name, destination)
        rows.append(
            {
                "path": name,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "catena-e26-stage3c-artifact-hash-manifest-v1",
        "manifest_type": "E26_STAGE3C_ARTIFACT_HASH_MANIFEST",
        "scientific_evidence": False,
        "predecessor_disposition": ("BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE"),
        "artifact_root": str(raw_root),
        "file_count": 11,
        "files": rows,
        "aggregate_algorithm": "path_nul_bytes_nul_sha256_newline_v1",
        "aggregate_sha256": _aggregate(rows),
        "source_commit": "3" * 40,
        "predecessor_mutated": False,
        "registered_predecessor": {
            "result": {"path": str(RESULT), "sha256": sha256_file(RESULT)},
            "status": {"path": str(STATUS), "sha256": sha256_file(STATUS)},
            "raw_run_aggregate_sha256": REGISTERED_RAW_AGGREGATE_SHA256,
            "failure_status_sha256": sha256_file(raw_root / "failure_status.json"),
            "disposition": ("BLOCKED_NUMERICAL_GRADIENT_ACCUMULATION_LAYOUT_INVARIANCE"),
        },
    }
    manifest["manifest_sha256"] = sha256_canonical_json(manifest)
    manifest_path = tmp_path / "stage3c_artifact_manifest.json"
    write_json_strict(manifest_path, manifest)

    protocol = tmp_path / "stage3c_protocol.json"
    write_json_strict(
        protocol,
        {
            "full_config_snapshot": {
                "backend_gates": {
                    "fp32_full_chunk_relative_l2_max": 1.0e-5,
                    "fp32_full_chunk_max_abs_max": 1.0e-5,
                    "bf16_fp32_relative_l2_max": 7.0e-3,
                    "gradient_norm_min": 1.0e-8,
                    "gradient_norm_max": "1.0e3",
                }
            }
        },
    )
    frozen = tmp_path / "frozen.json"
    write_json_strict(
        frozen,
        {
            "manifest_type": "E26_FROZEN_INVARIANCE_RECEIPT",
            "passed": True,
            "scientific_evidence": False,
        },
    )
    return {
        "result": RESULT,
        "status": STATUS,
        "protocol": protocol,
        "artifact": manifest_path,
        "raw_root": raw_root,
        "frozen": frozen,
    }
