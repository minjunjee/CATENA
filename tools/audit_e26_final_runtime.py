#!/usr/bin/env python3
"""Create a fail-closed E26 Final official-runtime dependency receipt.

This command performs metadata, filesystem, Git, compiler-version, and
``nvidia-smi`` inventory only.  It never calls a torch CUDA API, allocates a
CUDA tensor, launches a kernel, imports an ambient fallback backend, or starts
Scientific E26 Final.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any, Final

from catena.core.provenance_v61 import (
    loads_json_strict,
    read_json_object_strict,
    sha256_canonical_json,
    sha256_file,
    write_json_strict,
)


class E26FinalRuntimeAuditError(RuntimeError):
    """Raised when runtime provenance is malformed or cannot be audited safely."""


@dataclass(frozen=True)
class RuntimeExpectation:
    """Prospective exact dependency and hardware contract."""

    python_version: str = "3.11.15"
    torch_version: str = "2.9.0+cu130"
    torch_distribution_version: str = "2.9.0+cu130"
    torch_cuda_version: str = "13.0"
    flash_attn_version: str = "2.8.3"
    flash_attn_wheel_filename: str = (
        "flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl"
    )
    flash_attn_wheel_bytes: int = 239_981_365
    flash_attn_wheel_sha256: str = (
        "9f252bd59cfe19aef68434a87ba440ac6af35ab8f64d4440bda63b53667055cc"
    )
    flash_attn_wheel_tag: str = "cp311-cp311-linux_x86_64"
    fla_distribution_version: str = "0.5.1"
    # Newer FLA revisions removed API keywords that the pinned official
    # GDN2 source calls (notably ``use_exp2`` and
    # ``transpose_state_layout``).  This is the newest public parent commit
    # found before that incompatible API removal, and is therefore the one
    # prospective compatibility pin for E26 Final.
    fla_commit: str = "4b02d15d6a68700181b180235be62a9fb95d2a38"
    fla_tree: str = "816817b67e1bc3f8cd905f309034a1bd0d45b2da"
    official_commit: str = "95709fc250357c2dd109361c353192f2aa5913f9"
    official_tree: str = "bec1976e3b1ab0fab519f60c73e36a3c0092da47"
    official_gdn2_sha256: str = (
        "5d93765adcb4e9bf755e7d4160a01d4e2ee8438ec55759d17903223dd18b0324"
    )
    derived_gdn2_sha256: str = (
        "28476bc2c48d18b2548f2761b4beb2f9bf18ae63ae3c67cc4dc93fb81eab6ea8"
    )
    gate_patch_sha256: str = (
        "e94d546a80fcff0db7ffa08c1f0d2c0b650d3f912f4af729ee8ad61a993b2e33"
    )
    # The first completed receipt prospectively locks these environment bytes.
    # Supplying explicit values keeps repair/reproduction audits exact.
    pip_freeze_sha256: str | None = None
    pip_freeze_line_count: int | None = None
    nvcc_release: str = "13.0"
    nvcc_build: str = "13.0.88"
    gpu_model: str = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    driver_version: str = "580.126.16"
    required_gpu_indices: tuple[int, ...] = (0, 1, 2, 3)


DEFAULT_EXPECTATION: Final = RuntimeExpectation()
TARGET_GATE_SOURCE: Final = Path("lit_gpt/gdn2.py")
_RECEIPT_SCHEMA: Final = "catena-e26-final-official-runtime-v1"
_RECEIPT_TYPE: Final = "E26_FINAL_OFFICIAL_RUNTIME_RECEIPT"
_FROZEN_ENV: Final = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

_RUNTIME_PROBE = r"""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import pathlib
import platform
import sys

import torch

def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None

def distribution(name):
    dist = metadata.distribution(name)
    wheel = dist.read_text("WHEEL")
    direct_url = dist.read_text("direct_url.json")
    return {
        "version": dist.version,
        "dist_info": str(pathlib.Path(dist._path).resolve()),
        "wheel_sha256": digest_text(wheel),
        "metadata_sha256": digest_text(dist.read_text("METADATA")),
        "record_sha256": digest_text(dist.read_text("RECORD")),
        "installer": (dist.read_text("INSTALLER") or "").strip() or None,
        "direct_url": json.loads(direct_url) if direct_url is not None else None,
    }

def module_origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None or spec.origin is None else str(pathlib.Path(spec.origin).resolve())

payload = {
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(pathlib.Path(sys.executable).resolve()),
        "prefix": str(pathlib.Path(sys.prefix).resolve()),
        "base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
    },
    "torch": {
        "version": torch.__version__,
        "distribution_version": metadata.version("torch"),
        "cuda_version": torch.version.cuda,
        "module_origin": module_origin("torch"),
    },
    "distributions": {
        "flash_attn": distribution("flash-attn"),
        "flash_linear_attention": distribution("flash-linear-attention"),
    },
    "module_origins": {
        "flash_attn": module_origin("flash_attn"),
        "fla": module_origin("fla"),
    },
    "cuda_api_called": False,
    "cuda_tensor_allocated": False,
    "gpu_kernel_launched": False,
}
print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
"""


def _run(
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        tuple(arguments),
        check=False,
        capture_output=True,
        text=True,
        env=None if environment is None else dict(environment),
    )
    if completed.returncode != 0:
        raise E26FinalRuntimeAuditError(
            f"Command failed ({' '.join(arguments)}): {completed.stderr.strip()}"
        )
    return completed


def _git(repository: Path, *arguments: str, preserve_leading: bool = False) -> str:
    output = _run(("git", "-C", str(repository), *arguments)).stdout
    return output.rstrip("\n") if preserve_leading else output.strip()


def _real_file(
    path: str | Path,
    *,
    label: str,
    allow_executable_symlink: bool = False,
) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink() and not allow_executable_symlink:
        raise E26FinalRuntimeAuditError(f"{label} must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file():
        raise E26FinalRuntimeAuditError(f"{label} is not a regular file")
    return resolved


def _real_directory(path: str | Path, *, label: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise E26FinalRuntimeAuditError(f"{label} must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir():
        raise E26FinalRuntimeAuditError(f"{label} is not a directory")
    return resolved


def collect_python_runtime(python: str | Path) -> dict[str, Any]:
    """Collect CPU-only package metadata from the isolated official prefix."""

    executable = _real_file(
        python,
        label="Official Python executable",
        allow_executable_symlink=True,
    )
    environment = dict(os.environ)
    environment.update(_FROZEN_ENV)
    completed = _run(
        (str(executable), "-I", "-c", _RUNTIME_PROBE),
        environment=environment,
    )
    try:
        payload = loads_json_strict(completed.stdout)
    except ValueError as exc:
        raise E26FinalRuntimeAuditError("Official Python returned invalid strict JSON") from exc
    if not isinstance(payload, dict):
        raise E26FinalRuntimeAuditError("Official Python runtime root must be a JSON object")
    payload["probe_stderr_sha256"] = hashlib.sha256(completed.stderr.encode()).hexdigest()
    payload["probe_environment"] = dict(_FROZEN_ENV)
    return payload


def collect_pip_freeze(python: str | Path) -> tuple[bytes, bytes]:
    """Run pip freeze twice so the receipt binds deterministic environment bytes."""

    executable = _real_file(
        python,
        label="Official Python executable",
        allow_executable_symlink=True,
    )
    environment = dict(os.environ)
    environment.update(_FROZEN_ENV)
    command = (str(executable), "-I", "-m", "pip", "freeze", "--all")
    first = _run(command, environment=environment).stdout.encode("utf-8")
    second = _run(command, environment=environment).stdout.encode("utf-8")
    return first, second


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise E26FinalRuntimeAuditError(f"{label} must be a string-keyed mapping")
    return value


def audit_python_runtime(
    observation: Mapping[str, Any],
    *,
    expected_executable: str | Path,
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Audit Python, torch/CUDA metadata, installed wheels, and FLA origin."""

    python = _mapping(observation.get("python"), "runtime.python")
    torch_row = _mapping(observation.get("torch"), "runtime.torch")
    distributions = _mapping(observation.get("distributions"), "runtime.distributions")
    flash = _mapping(distributions.get("flash_attn"), "flash-attn distribution")
    fla = _mapping(
        distributions.get("flash_linear_attention"),
        "flash-linear-attention distribution",
    )
    module_origins = _mapping(observation.get("module_origins"), "module origins")
    executable = Path(expected_executable).expanduser().resolve(strict=True)
    prefix = executable.parent.parent
    checks = {
        "python_executable_exact": python.get("executable") == str(executable),
        "python_prefix_exact": python.get("prefix") == str(prefix),
        "python_implementation_cpython": python.get("implementation") == "CPython",
        "python_version_exact": python.get("version") == expected.python_version,
        "torch_version_exact": torch_row.get("version") == expected.torch_version,
        "torch_distribution_version_exact": torch_row.get("distribution_version")
        == expected.torch_distribution_version,
        "torch_cuda_version_exact": torch_row.get("cuda_version")
        == expected.torch_cuda_version,
        "flash_attn_distribution_version_exact": flash.get("version")
        == expected.flash_attn_version,
        "flash_attn_installed_by_pip": flash.get("installer") == "pip",
        "flash_attn_module_within_prefix": _path_is_within(
            module_origins.get("flash_attn"), prefix
        ),
        "fla_distribution_version_exact": fla.get("version")
        == expected.fla_distribution_version,
        "no_torch_cuda_api_called": observation.get("cuda_api_called") is False,
        "no_cuda_tensor_allocated": observation.get("cuda_tensor_allocated") is False,
        "no_gpu_kernel_launched": observation.get("gpu_kernel_launched") is False,
    }
    return {
        "observation": deepcopy(dict(observation)),
        "environment_prefix": str(prefix),
        "fla_direct_url": deepcopy(fla.get("direct_url")),
        "fla_module_origin": module_origins.get("fla"),
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _path_is_within(raw: object, root: Path) -> bool:
    if not isinstance(raw, str):
        return False
    try:
        return Path(raw).resolve(strict=True).is_relative_to(root)
    except (FileNotFoundError, OSError):
        return False


def _wheel_member(archive: zipfile.ZipFile, suffix: str) -> tuple[str, bytes]:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise E26FinalRuntimeAuditError(
            f"Wheel must contain exactly one {suffix}, observed {len(names)}"
        )
    return names[0], archive.read(names[0])


def audit_flash_attn_wheel(
    wheel_path: str | Path,
    *,
    runtime_section: Mapping[str, Any],
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Bind the exact cached flash-attn wheel to installed dist metadata."""

    wheel = _real_file(wheel_path, label="flash-attn wheel")
    with zipfile.ZipFile(wheel) as archive:
        metadata_name, metadata_bytes = _wheel_member(archive, ".dist-info/METADATA")
        wheel_name, wheel_bytes = _wheel_member(archive, ".dist-info/WHEEL")
    parsed = Parser().parsestr(metadata_bytes.decode("utf-8"))
    tags = [
        line.partition(":")[2].strip()
        for line in wheel_bytes.decode("utf-8").splitlines()
        if line.startswith("Tag:")
    ]
    observation = _mapping(runtime_section.get("observation"), "runtime observation")
    distributions = _mapping(observation.get("distributions"), "runtime distributions")
    installed = _mapping(distributions.get("flash_attn"), "installed flash-attn")
    observed_sha256 = sha256_file(wheel)
    checks = {
        "wheel_filename_exact": wheel.name == expected.flash_attn_wheel_filename,
        "wheel_bytes_exact": wheel.stat().st_size == expected.flash_attn_wheel_bytes,
        "wheel_sha256_exact": observed_sha256 == expected.flash_attn_wheel_sha256,
        "wheel_distribution_name_exact": parsed.get("Name") == "flash_attn",
        "wheel_version_exact": parsed.get("Version") == expected.flash_attn_version,
        "wheel_tag_exact": tags == [expected.flash_attn_wheel_tag],
        "installed_version_matches_wheel": installed.get("version")
        == parsed.get("Version"),
        "installed_metadata_matches_wheel": installed.get("metadata_sha256")
        == hashlib.sha256(metadata_bytes).hexdigest(),
        "installed_wheel_metadata_matches_archive": installed.get("wheel_sha256")
        == hashlib.sha256(wheel_bytes).hexdigest(),
    }
    return {
        "path": str(wheel),
        "bytes": wheel.stat().st_size,
        "sha256": observed_sha256,
        "metadata_member": metadata_name,
        "wheel_member": wheel_name,
        "distribution_name": parsed.get("Name"),
        "version": parsed.get("Version"),
        "tags": tags,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def audit_pip_freeze(
    first: bytes,
    second: bytes,
    *,
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Bind exact, replay-stable ``pip freeze --all`` bytes."""

    digest = hashlib.sha256(first).hexdigest()
    lines = first.decode("utf-8").splitlines()
    checks = {
        "pip_freeze_replay_exact": first == second,
        "pip_freeze_sha256_bound": (
            len(digest) == 64
            if expected.pip_freeze_sha256 is None
            else digest == expected.pip_freeze_sha256
        ),
        "pip_freeze_line_count_bound": (
            bool(lines)
            if expected.pip_freeze_line_count is None
            else len(lines) == expected.pip_freeze_line_count
        ),
        "pip_freeze_nonempty_unique_lines": bool(lines) and len(lines) == len(set(lines)),
    }
    return {
        "sha256": digest,
        "bytes": len(first),
        "line_count": len(lines),
        "lines": lines,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _git_checkout(
    path: str | Path,
    *,
    expected_commit: str,
    expected_tree: str,
    clean: bool,
) -> dict[str, Any]:
    root = _real_directory(path, label="Git checkout")
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        preserve_leading=True,
    )
    return {
        "path": str(root),
        "head": head,
        "tree": tree,
        "status_porcelain": status.splitlines(),
        "remote_url": _git(root, "remote", "get-url", "origin"),
        "commit_exact": head == expected_commit,
        "tree_exact": tree == expected_tree,
        "clean_exact": (status == "") is clean,
    }


def audit_fla_source(
    fla_source: str | Path,
    *,
    runtime_section: Mapping[str, Any],
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Bind clean FLA Git bytes, editable receipt, and actual import origin."""

    checkout = _git_checkout(
        fla_source,
        expected_commit=expected.fla_commit,
        expected_tree=expected.fla_tree,
        clean=True,
    )
    root = Path(checkout["path"])
    direct_url = runtime_section.get("fla_direct_url")
    expected_url = root.as_uri()
    direct_url_exact = (
        isinstance(direct_url, Mapping)
        and direct_url.get("url") == expected_url
        and isinstance(direct_url.get("dir_info"), Mapping)
        and direct_url["dir_info"].get("editable") is True
    )
    origin_within = _path_is_within(runtime_section.get("fla_module_origin"), root)
    checks = {
        "fla_commit_exact": bool(checkout["commit_exact"]),
        "fla_tree_exact": bool(checkout["tree_exact"]),
        "fla_checkout_clean": checkout["status_porcelain"] == [],
        "fla_editable_direct_url_exact": direct_url_exact,
        "fla_import_origin_within_clean_checkout": origin_within,
    }
    return {
        "checkout": checkout,
        "expected_editable_url": expected_url,
        "observed_direct_url": deepcopy(direct_url),
        "observed_module_origin": runtime_section.get("fla_module_origin"),
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def _render_gate_diff(base: bytes, derived: bytes) -> bytes:
    before = base.decode("utf-8").splitlines(keepends=True)
    after = derived.decode("utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=f"a/{TARGET_GATE_SOURCE.as_posix()}",
            tofile=f"b/{TARGET_GATE_SOURCE.as_posix()}",
        )
    ).encode("utf-8")


def audit_official_source_lineage(
    *,
    official_base: str | Path,
    derived_source: str | Path,
    gate_patch: str | Path,
    gate_patch_receipt: str | Path,
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Bind a clean base and its sole, exact gate-policy derived diff."""

    base = _git_checkout(
        official_base,
        expected_commit=expected.official_commit,
        expected_tree=expected.official_tree,
        clean=True,
    )
    derived = _git_checkout(
        derived_source,
        expected_commit=expected.official_commit,
        expected_tree=expected.official_tree,
        clean=False,
    )
    base_root = Path(base["path"])
    derived_root = Path(derived["path"])
    base_file = _real_file(base_root / TARGET_GATE_SOURCE, label="Official base gate source")
    derived_file = _real_file(
        derived_root / TARGET_GATE_SOURCE,
        label="Derived gate source",
    )
    patch_file = _real_file(gate_patch, label="Gate patch")
    patch_receipt_file = _real_file(gate_patch_receipt, label="Gate patch receipt")
    patch_bytes = patch_file.read_bytes()
    rendered = _render_gate_diff(base_file.read_bytes(), derived_file.read_bytes())
    receipt = read_json_object_strict(patch_receipt_file)
    changed_paths = _git(derived_root, "diff", "--name-only").splitlines()
    expected_status = [f" M {TARGET_GATE_SOURCE.as_posix()}"]
    checks = {
        "official_base_commit_exact": bool(base["commit_exact"]),
        "official_base_tree_exact": bool(base["tree_exact"]),
        "official_base_clean": base["status_porcelain"] == [],
        "official_base_gdn2_sha256_exact": sha256_file(base_file)
        == expected.official_gdn2_sha256,
        "derived_base_commit_exact": bool(derived["commit_exact"]),
        "derived_base_tree_exact": bool(derived["tree_exact"]),
        "derived_only_gate_source_modified": derived["status_porcelain"]
        == expected_status
        and changed_paths == [TARGET_GATE_SOURCE.as_posix()],
        "derived_gdn2_sha256_exact": sha256_file(derived_file)
        == expected.derived_gdn2_sha256,
        "gate_patch_bytes_exact": patch_bytes == rendered,
        "gate_patch_sha256_exact": sha256_file(patch_file)
        == expected.gate_patch_sha256,
        "patch_receipt_base_sha_exact": receipt.get("base_file_sha256")
        == expected.official_gdn2_sha256,
        "patch_receipt_derived_sha_exact": receipt.get("patched_file_sha256")
        == expected.derived_gdn2_sha256,
        "patch_receipt_diff_sha_exact": receipt.get("unified_diff_sha256")
        == expected.gate_patch_sha256,
        "patch_receipt_commit_exact": receipt.get("official_commit")
        == expected.official_commit,
        "patch_receipt_gate_only": receipt.get("kernel_calls_modified") is False
        and receipt.get("target_relative_path") == TARGET_GATE_SOURCE.as_posix(),
        "patch_receipt_policy_contract_exact": receipt.get("status") == "APPLIED"
        and receipt.get("explicit_policy_required") is True
        and receipt.get("allowed_policy_values")
        == ["dual_gdn2", "projected_tied_gdn2"]
        and receipt.get("projection_heads_preserved") == ["b_proj", "w_proj"],
    }
    return {
        "official_base": base,
        "derived_source": derived,
        "base_gate_source_sha256": sha256_file(base_file),
        "derived_gate_source_sha256": sha256_file(derived_file),
        "gate_patch": {
            "path": str(patch_file),
            "bytes": patch_file.stat().st_size,
            "sha256": sha256_file(patch_file),
        },
        "gate_patch_receipt": {
            "path": str(patch_receipt_file),
            "sha256": sha256_file(patch_receipt_file),
            "payload": receipt,
        },
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def audit_nvcc(
    nvcc_path: str | Path,
    version_output: str,
    *,
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Bind the exact nvcc binary and parsed CUDA toolkit release."""

    nvcc = _real_file(
        nvcc_path,
        label="nvcc",
        allow_executable_symlink=True,
    )
    release = re.search(r"release\s+([0-9.]+),\s+V([0-9.]+)", version_output)
    observed_release = release.group(1) if release is not None else None
    observed_build = release.group(2) if release is not None else None
    checks = {
        "nvcc_output_parseable": release is not None,
        "nvcc_release_exact": observed_release == expected.nvcc_release,
        "nvcc_build_exact": observed_build == expected.nvcc_build,
    }
    return {
        "path": str(nvcc),
        "bytes": nvcc.stat().st_size,
        "sha256": sha256_file(nvcc),
        "version_output": version_output.splitlines(),
        "release": observed_release,
        "build": observed_build,
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def parse_gpu_inventory(output: str) -> list[dict[str, Any]]:
    """Parse no-header nounits nvidia-smi inventory output."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise E26FinalRuntimeAuditError(
                f"Malformed nvidia-smi row {line_number}: expected five fields"
            )
        try:
            index = int(fields[0])
            memory_mib = int(fields[4])
        except ValueError as exc:
            raise E26FinalRuntimeAuditError(
                f"Malformed numeric nvidia-smi row {line_number}"
            ) from exc
        rows.append(
            {
                "index": index,
                "name": fields[1],
                "uuid": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": memory_mib,
            }
        )
    if len({row["index"] for row in rows}) != len(rows):
        raise E26FinalRuntimeAuditError("nvidia-smi returned duplicate GPU indices")
    return rows


def audit_gpu_inventory(
    rows: Sequence[Mapping[str, Any]],
    *,
    nvidia_smi_path: str | Path,
    expected: RuntimeExpectation = DEFAULT_EXPECTATION,
) -> dict[str, Any]:
    """Bind selected physical GPU models and driver without launching a kernel."""

    executable = _real_file(
        nvidia_smi_path,
        label="nvidia-smi",
        allow_executable_symlink=True,
    )
    by_index = {row.get("index"): row for row in rows}
    selected = [by_index.get(index) for index in expected.required_gpu_indices]
    selected_present = all(row is not None for row in selected)
    selected_rows = [row for row in selected if row is not None]
    checks = {
        "required_gpu_indices_present": selected_present,
        "selected_gpu_models_exact": selected_present
        and all(row.get("name") == expected.gpu_model for row in selected_rows),
        "selected_driver_versions_exact": selected_present
        and all(
            row.get("driver_version") == expected.driver_version for row in selected_rows
        ),
        "selected_gpu_uuids_present": selected_present
        and all(
            isinstance(row.get("uuid"), str) and bool(row.get("uuid"))
            for row in selected_rows
        ),
        "selected_gpu_memory_reported": selected_present
        and all(
            isinstance(row.get("memory_total_mib"), int)
            and not isinstance(row.get("memory_total_mib"), bool)
            and row["memory_total_mib"] > 0
            for row in selected_rows
        ),
        "gpu_compute_not_executed": True,
    }
    return {
        "nvidia_smi": {
            "path": str(executable),
            "bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "inventory": [deepcopy(dict(row)) for row in rows],
        "selected_indices": list(expected.required_gpu_indices),
        "hard_checks": checks,
        "passed": all(checks.values()),
    }


def build_runtime_receipt(
    *,
    python_runtime: Mapping[str, Any],
    flash_attn_wheel: Mapping[str, Any],
    pip_freeze: Mapping[str, Any],
    fla_source: Mapping[str, Any],
    official_source_lineage: Mapping[str, Any],
    cuda_toolkit: Mapping[str, Any],
    gpu_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic receipt while keeping decode plumbing unresolved."""

    sections = {
        "python_runtime": deepcopy(dict(python_runtime)),
        "flash_attn_wheel": deepcopy(dict(flash_attn_wheel)),
        "pip_freeze": deepcopy(dict(pip_freeze)),
        "fla_source": deepcopy(dict(fla_source)),
        "official_source_lineage": deepcopy(dict(official_source_lineage)),
        "cuda_toolkit": deepcopy(dict(cuda_toolkit)),
        "gpu_inventory": deepcopy(dict(gpu_inventory)),
    }
    checks: dict[str, bool] = {}
    for section_name, section in sections.items():
        section_checks = section.get("hard_checks")
        if not isinstance(section_checks, Mapping) or not section_checks or not all(
            isinstance(value, bool) for value in section_checks.values()
        ):
            raise E26FinalRuntimeAuditError(
                f"{section_name} lacks non-empty boolean hard checks"
            )
        section_passed = all(section_checks.values())
        if section.get("passed") is not section_passed:
            raise E26FinalRuntimeAuditError(
                f"{section_name} disposition differs from its hard checks"
            )
        checks.update(
            {f"{section_name}.{key}": value for key, value in section_checks.items()}
        )
    passed = all(checks.values())
    limitation = {
        "code": "EXTERNAL_DECODE_CACHE_PLUMBING_NOT_IMPLEMENTED",
        "status": "KNOWN_UNRESOLVED_LIMITATION",
        "protocol_hard_gate_for_dependency_receipt": False,
        "scientific_decode_gate_open": False,
        "detail": (
            "The official GPT owns decode/recurrent caches internally and the E26 Final "
            "gate-only derived source does not provide an externally serializable, cloneable, "
            "restorable cache adapter. This receipt does not establish query-branch state "
            "cloning or external decode-cache correctness."
        ),
    }
    receipt: dict[str, Any] = {
        "schema_version": _RECEIPT_SCHEMA,
        "manifest_type": _RECEIPT_TYPE,
        "scientific_evidence": False,
        "evidence_tier": "NON_EVIDENCE_RUNTIME_DEPENDENCY_PROVENANCE",
        "claim_ceiling": "OFFICIAL_RUNTIME_DEPENDENCY_ADMISSION_ONLY",
        "gpu_execution": {
            "torch_cuda_api_called": False,
            "cuda_tensor_allocated": False,
            "gpu_kernel_launched": False,
            "nvidia_smi_inventory_only": True,
        },
        **sections,
        "protocol_hard_checks": checks,
        "limitations": [limitation],
        "external_decode_cache_plumbing_implemented": False,
        "decode_cache_evaluation_eligible": False,
        "runtime_dependency_eligible": passed,
        "scientific_e26_final_execution_eligible": False,
        "scientific_e26a_started": False,
        "passed": passed,
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return validate_runtime_receipt(receipt)


def validate_runtime_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable hash, hard checks, and mandatory cache limitation."""

    normalized = deepcopy(dict(payload))
    claimed = normalized.pop("receipt_sha256", None)
    if claimed != sha256_canonical_json(normalized):
        raise E26FinalRuntimeAuditError("Runtime receipt SHA-256 changed")
    normalized["receipt_sha256"] = claimed
    if (
        normalized.get("schema_version") != _RECEIPT_SCHEMA
        or normalized.get("manifest_type") != _RECEIPT_TYPE
        or normalized.get("scientific_evidence") is not False
        or normalized.get("scientific_e26_final_execution_eligible") is not False
        or normalized.get("scientific_e26a_started") is not False
    ):
        raise E26FinalRuntimeAuditError("Runtime receipt evidence boundary changed")
    checks = normalized.get("protocol_hard_checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise E26FinalRuntimeAuditError("Runtime receipt hard checks are invalid")
    for section_name in (
        "python_runtime",
        "flash_attn_wheel",
        "pip_freeze",
        "fla_source",
        "official_source_lineage",
        "cuda_toolkit",
        "gpu_inventory",
    ):
        section = normalized.get(section_name)
        if not isinstance(section, Mapping):
            raise E26FinalRuntimeAuditError(f"Runtime receipt lacks {section_name}")
        section_checks = section.get("hard_checks")
        if not isinstance(section_checks, Mapping) or section.get("passed") is not all(
            section_checks.values()
        ):
            raise E26FinalRuntimeAuditError(
                f"{section_name} disposition differs from hard checks"
            )
    limitations = normalized.get("limitations")
    required_code = "EXTERNAL_DECODE_CACHE_PLUMBING_NOT_IMPLEMENTED"
    if (
        not isinstance(limitations, list)
        or len(limitations) != 1
        or not isinstance(limitations[0], Mapping)
        or limitations[0].get("code") != required_code
        or limitations[0].get("scientific_decode_gate_open") is not False
        or normalized.get("external_decode_cache_plumbing_implemented") is not False
        or normalized.get("decode_cache_evaluation_eligible") is not False
    ):
        raise E26FinalRuntimeAuditError("External decode-cache limitation is missing")
    expected_pass = all(checks.values())
    if normalized.get("passed") is not expected_pass or normalized.get(
        "runtime_dependency_eligible"
    ) is not expected_pass:
        raise E26FinalRuntimeAuditError("Runtime receipt disposition is inconsistent")
    return normalized


def write_runtime_receipt(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one immutable strict-JSON runtime receipt."""

    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite runtime receipt: {destination}")
    validated = validate_runtime_receipt(payload)
    write_json_strict(destination, validated)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E26 Final official runtime dependencies without GPU execution"
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/envs/gdn2_official_95709fc/bin/python"),
    )
    parser.add_argument("--flash-attn-wheel", type=Path, required=True)
    parser.add_argument(
        "--fla-source",
        type=Path,
        default=Path(
            "/data/minjun_dev/CATENA/official_sources/"
            "fla_gdn2_api_compat_4b02d15"
        ),
    )
    parser.add_argument(
        "--official-base",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/external/gdn2_official"),
    )
    parser.add_argument(
        "--derived-source",
        type=Path,
        default=Path("/data/minjun_dev/CATENA/external/gdn2_e26_final_runtime"),
    )
    parser.add_argument("--gate-patch", type=Path, required=True)
    parser.add_argument("--gate-patch-receipt", type=Path, required=True)
    parser.add_argument(
        "--nvcc",
        type=Path,
        default=Path("/usr/local/cuda-13.0/bin/nvcc"),
    )
    parser.add_argument(
        "--nvidia-smi",
        type=Path,
        default=Path("/usr/bin/nvidia-smi"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime_observation = collect_python_runtime(args.python)
    runtime = audit_python_runtime(
        runtime_observation,
        expected_executable=args.python,
    )
    freeze_first, freeze_second = collect_pip_freeze(args.python)
    wheel = audit_flash_attn_wheel(
        args.flash_attn_wheel,
        runtime_section=runtime,
    )
    freeze = audit_pip_freeze(freeze_first, freeze_second)
    fla = audit_fla_source(args.fla_source, runtime_section=runtime)
    lineage = audit_official_source_lineage(
        official_base=args.official_base,
        derived_source=args.derived_source,
        gate_patch=args.gate_patch,
        gate_patch_receipt=args.gate_patch_receipt,
    )
    nvcc_path = _real_file(
        args.nvcc,
        label="nvcc",
        allow_executable_symlink=True,
    )
    nvcc_output = _run((str(nvcc_path), "--version")).stdout
    cuda = audit_nvcc(nvcc_path, nvcc_output)
    nvidia_smi = _real_file(
        args.nvidia_smi,
        label="nvidia-smi",
        allow_executable_symlink=True,
    )
    inventory_output = _run(
        (
            str(nvidia_smi),
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    ).stdout
    gpu = audit_gpu_inventory(
        parse_gpu_inventory(inventory_output),
        nvidia_smi_path=nvidia_smi,
    )
    receipt = build_runtime_receipt(
        python_runtime=runtime,
        flash_attn_wheel=wheel,
        pip_freeze=freeze,
        fla_source=fla,
        official_source_lineage=lineage,
        cuda_toolkit=cuda,
        gpu_inventory=gpu,
    )
    output = write_runtime_receipt(args.output, receipt)
    print(f"E26 Final official runtime dependencies: {'PASS' if receipt['passed'] else 'BLOCKED'}")
    print(f"receipt: {output.resolve()}")
    print(f"receipt_sha256: {receipt['receipt_sha256']}")
    print("external_decode_cache_plumbing: KNOWN_UNRESOLVED_LIMITATION")
    print("scientific_e26_final_execution_eligible: false")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
