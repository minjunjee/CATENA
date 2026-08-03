from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from catena.core.provenance_v61 import read_json_object_strict
from tools import audit_e26_final_runtime as runtime_audit
from tools.audit_e26_final_runtime import (
    DEFAULT_EXPECTATION,
    E26FinalRuntimeAuditError,
    RuntimeExpectation,
    audit_fla_source,
    audit_flash_attn_wheel,
    audit_gpu_inventory,
    audit_nvcc,
    audit_official_source_lineage,
    audit_pip_freeze,
    audit_python_runtime,
    build_runtime_receipt,
    parse_gpu_inventory,
    validate_runtime_receipt,
    write_runtime_receipt,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(root: Path, files: dict[str, bytes]) -> tuple[str, str]:
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(root, "init", "-b", "fixture")
    _git(root, "config", "user.name", "CATENA Test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "remote", "add", "origin", "https://example.invalid/source.git")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def _runtime_observation(
    env_root: Path,
    fla_root: Path,
    *,
    expected: RuntimeExpectation,
) -> dict[str, Any]:
    python = env_root / "bin" / "python3.11"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"python-fixture")
    flash_module = env_root / "lib" / "python3.11" / "site-packages" / "flash_attn"
    flash_module.mkdir(parents=True)
    flash_init = flash_module / "__init__.py"
    flash_init.write_text("", encoding="utf-8")
    fla_init = fla_root / "fla" / "__init__.py"
    fla_init.parent.mkdir(parents=True, exist_ok=True)
    if not fla_init.exists():
        fla_init.write_text("", encoding="utf-8")
    return {
        "python": {
            "version": expected.python_version,
            "implementation": "CPython",
            "executable": str(python.resolve()),
            "prefix": str(env_root.resolve()),
            "base_prefix": str(env_root.resolve()),
        },
        "torch": {
            "version": expected.torch_version,
            "distribution_version": expected.torch_distribution_version,
            "cuda_version": expected.torch_cuda_version,
            "module_origin": str(
                env_root / "lib" / "python3.11" / "site-packages" / "torch" / "__init__.py"
            ),
        },
        "distributions": {
            "flash_attn": {
                "version": expected.flash_attn_version,
                "installer": "pip",
                "metadata_sha256": None,
                "wheel_sha256": None,
                "record_sha256": "fixture",
                "direct_url": None,
            },
            "flash_linear_attention": {
                "version": expected.fla_distribution_version,
                "installer": "pip",
                "metadata_sha256": "fixture",
                "wheel_sha256": "fixture",
                "record_sha256": "fixture",
                "direct_url": {
                    "dir_info": {"editable": True},
                    "url": fla_root.resolve().as_uri(),
                },
            },
        },
        "module_origins": {
            "flash_attn": str(flash_init.resolve()),
            "fla": str(fla_init.resolve()),
        },
        "cuda_api_called": False,
        "cuda_tensor_allocated": False,
        "gpu_kernel_launched": False,
    }


def _runtime_section(
    tmp_path: Path,
    fla_root: Path,
    *,
    expected: RuntimeExpectation,
) -> tuple[dict[str, Any], Path]:
    env_root = tmp_path / "env"
    observation = _runtime_observation(env_root, fla_root, expected=expected)
    python = env_root / "bin" / "python3.11"
    return (
        audit_python_runtime(
            observation,
            expected_executable=python,
            expected=expected,
        ),
        python,
    )


def _make_wheel(path: Path, version: str) -> tuple[bytes, bytes]:
    metadata = f"Metadata-Version: 2.1\nName: flash_attn\nVersion: {version}\n\n".encode()
    wheel = (
        b"Wheel-Version: 1.0\n"
        b"Generator: CATENA test\n"
        b"Root-Is-Purelib: false\n"
        b"Tag: cp311-cp311-linux_x86_64\n\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"flash_attn-{version}.dist-info/METADATA", metadata)
        archive.writestr(f"flash_attn-{version}.dist-info/WHEEL", wheel)
    return metadata, wheel


def _passed_section() -> dict[str, Any]:
    return {"hard_checks": {"bound": True}, "passed": True}


def test_python_runtime_binds_python_torch_cuda_and_package_origins(tmp_path: Path) -> None:
    fla_root = tmp_path / "fla"
    expected = replace(DEFAULT_EXPECTATION, pip_freeze_line_count=2)
    report, _python = _runtime_section(tmp_path, fla_root, expected=expected)
    assert report["passed"] is True
    assert all(report["hard_checks"].values())
    assert report["observation"]["torch"]["cuda_version"] == "13.0"
    assert report["observation"]["cuda_api_called"] is False


def test_flash_attn_wheel_binds_exact_bytes_version_tag_and_install_metadata(
    tmp_path: Path,
) -> None:
    wheel_path = tmp_path / "flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl"
    metadata, wheel_metadata = _make_wheel(wheel_path, "2.8.3")
    expected = replace(
        DEFAULT_EXPECTATION,
        flash_attn_wheel_bytes=wheel_path.stat().st_size,
        flash_attn_wheel_sha256=hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
    )
    runtime = {
        "observation": {
            "distributions": {
                "flash_attn": {
                    "version": "2.8.3",
                    "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
                    "wheel_sha256": hashlib.sha256(wheel_metadata).hexdigest(),
                }
            }
        }
    }
    report = audit_flash_attn_wheel(
        wheel_path,
        runtime_section=runtime,
        expected=expected,
    )
    assert report["passed"] is True
    assert report["tags"] == ["cp311-cp311-linux_x86_64"]


def test_pip_freeze_requires_replay_exact_locked_sha_and_line_count() -> None:
    payload = b"flash_attn==2.8.3\ntorch==2.9.0+cu130\n"
    expected = replace(
        DEFAULT_EXPECTATION,
        pip_freeze_sha256=hashlib.sha256(payload).hexdigest(),
        pip_freeze_line_count=2,
    )
    report = audit_pip_freeze(payload, payload, expected=expected)
    assert report["passed"] is True
    changed = audit_pip_freeze(payload, payload + b"x\n", expected=expected)
    assert changed["passed"] is False
    assert changed["hard_checks"]["pip_freeze_replay_exact"] is False


def test_fla_receipt_requires_clean_api_compatible_checkout_and_actual_import_origin(
    tmp_path: Path,
) -> None:
    fla_root = tmp_path / "fla"
    head, tree = _init_repo(fla_root, {"fla/__init__.py": b"version = 'fixture'\n"})
    expected = replace(
        DEFAULT_EXPECTATION,
        fla_commit=head,
        fla_tree=tree,
    )
    runtime, _python = _runtime_section(tmp_path, fla_root, expected=expected)
    report = audit_fla_source(fla_root, runtime_section=runtime, expected=expected)
    assert report["passed"] is True

    drifted = deepcopy(runtime)
    drifted["fla_module_origin"] = str(tmp_path / "old-fla" / "fla" / "__init__.py")
    mismatch = audit_fla_source(fla_root, runtime_section=drifted, expected=expected)
    assert mismatch["passed"] is False
    assert mismatch["hard_checks"]["fla_import_origin_within_clean_checkout"] is False


def test_official_lineage_binds_clean_base_and_single_exact_gate_diff(
    tmp_path: Path,
) -> None:
    base_bytes = b"def forward(x):\n    b = sigmoid(x)\n    w = sigmoid(x)\n"
    derived_bytes = (
        b"def forward(x):\n    b_logits = x\n    w_logits = x\n    b = sigmoid(b_logits)\n"
        b"    w = sigmoid(w_logits)\n"
    )
    base = tmp_path / "base"
    head, tree = _init_repo(base, {"lit_gpt/gdn2.py": base_bytes, "LICENSE": b"license\n"})
    derived = tmp_path / "derived"
    _git(tmp_path, "clone", str(base), str(derived))
    (derived / "lit_gpt" / "gdn2.py").write_bytes(derived_bytes)
    patch_bytes = runtime_audit._render_gate_diff(base_bytes, derived_bytes)
    patch = tmp_path / "gate.patch"
    patch.write_bytes(patch_bytes)
    expected = replace(
        DEFAULT_EXPECTATION,
        official_commit=head,
        official_tree=tree,
        official_gdn2_sha256=hashlib.sha256(base_bytes).hexdigest(),
        derived_gdn2_sha256=hashlib.sha256(derived_bytes).hexdigest(),
        gate_patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
    )
    patch_receipt = tmp_path / "patch.json"
    patch_receipt.write_text(
        json.dumps(
            {
                "status": "APPLIED",
                "official_commit": head,
                "base_file_sha256": expected.official_gdn2_sha256,
                "patched_file_sha256": expected.derived_gdn2_sha256,
                "unified_diff_sha256": expected.gate_patch_sha256,
                "kernel_calls_modified": False,
                "target_relative_path": "lit_gpt/gdn2.py",
                "explicit_policy_required": True,
                "allowed_policy_values": ["dual_gdn2", "projected_tied_gdn2"],
                "projection_heads_preserved": ["b_proj", "w_proj"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = audit_official_source_lineage(
        official_base=base,
        derived_source=derived,
        gate_patch=patch,
        gate_patch_receipt=patch_receipt,
        expected=expected,
    )
    assert report["passed"] is True
    assert report["derived_source"]["status_porcelain"] == [" M lit_gpt/gdn2.py"]

    (base / "LICENSE").write_text("dirty\n", encoding="utf-8")
    blocked = audit_official_source_lineage(
        official_base=base,
        derived_source=derived,
        gate_patch=patch,
        gate_patch_receipt=patch_receipt,
        expected=expected,
    )
    assert blocked["passed"] is False
    assert blocked["hard_checks"]["official_base_clean"] is False


def test_nvcc_and_gpu_inventory_are_metadata_only(tmp_path: Path) -> None:
    nvcc = tmp_path / "nvcc"
    nvcc.write_bytes(b"nvcc fixture")
    nvcc_report = audit_nvcc(
        nvcc,
        "Cuda compilation tools, release 13.0, V13.0.88\n",
    )
    assert nvcc_report["passed"] is True
    smi = tmp_path / "nvidia-smi"
    smi.write_bytes(b"smi fixture")
    output = "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-{index}, "
        "580.126.16, 97887"
        for index in range(4)
    )
    rows = parse_gpu_inventory(output)
    report = audit_gpu_inventory(rows, nvidia_smi_path=smi)
    assert report["passed"] is True
    assert report["hard_checks"]["gpu_compute_not_executed"] is True


def test_runtime_receipt_makes_external_decode_cache_limitation_unavoidable(
    tmp_path: Path,
) -> None:
    receipt = build_runtime_receipt(
        python_runtime=_passed_section(),
        flash_attn_wheel=_passed_section(),
        pip_freeze=_passed_section(),
        fla_source=_passed_section(),
        official_source_lineage=_passed_section(),
        cuda_toolkit=_passed_section(),
        gpu_inventory=_passed_section(),
    )
    assert receipt["passed"] is True
    assert receipt["runtime_dependency_eligible"] is True
    assert receipt["external_decode_cache_plumbing_implemented"] is False
    assert receipt["decode_cache_evaluation_eligible"] is False
    assert receipt["scientific_e26_final_execution_eligible"] is False
    assert receipt["limitations"][0]["code"] == (
        "EXTERNAL_DECODE_CACHE_PLUMBING_NOT_IMPLEMENTED"
    )

    output = tmp_path / "runtime_receipt.json"
    write_runtime_receipt(output, receipt)
    assert read_json_object_strict(output) == receipt
    with pytest.raises(FileExistsError):
        write_runtime_receipt(output, receipt)

    tampered = deepcopy(receipt)
    tampered["external_decode_cache_plumbing_implemented"] = True
    with pytest.raises(E26FinalRuntimeAuditError, match="SHA-256"):
        validate_runtime_receipt(tampered)


def test_probe_source_contains_no_torch_cuda_call_or_kernel_launch() -> None:
    assert "torch.cuda" not in runtime_audit._RUNTIME_PROBE
    assert "gpu_kernel_launched\": False" in runtime_audit._RUNTIME_PROBE
