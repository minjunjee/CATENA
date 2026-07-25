from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
BLOCKED = "BLOCKED"
ERROR = "ERROR"
HARD_FAILURE_STATUSES = {FAIL, BLOCKED, ERROR}

SUPPORTED_CHECKS = {
    "conda_environment",
    "python_runtime",
    "python_packages",
    "host_gpu_inventory",
    "cuda_toolchain",
    "cuda_bf16_lanes",
    "pytorch_cuda",
    "pytorch_bf16_lanes",
    "state_cache_storage",
    "repository_validation",
    "reproducibility_manifest",
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "output_dir",
    "checks",
    "environment",
    "hardware",
    "toolchain",
    "storage",
    "repository",
}

ENVIRONMENT_KEYS = {
    "conda_name",
    "python_major_minor",
    "torch_version",
    "torch_cuda_runtime",
    "driver_version",
    "required_packages",
    "minimum_glibc",
}
HARDWARE_KEYS = {
    "selected_physical_gpus",
    "expected_visible_gpu_count",
    "expected_model",
    "minimum_memory_mib",
    "compute_capability",
    "mig_mode",
}
TOOLCHAIN_KEYS = {
    "cuda_toolkit_release",
    "compile_timeout_seconds",
    "lane_timeout_seconds",
}
STORAGE_KEYS = {
    "path",
    "model_cache_path",
    "probe_size_mib",
    "repeats",
    "recommended_free_gib",
}
REPOSITORY_KEYS = {"run_pytest", "run_config_audit", "run_mock_smoke"}

TORCH_LANE_PROBE = r"""
import json
import math
import os
import subprocess
import time

import torch

identity_process = subprocess.run(
    [os.environ["CATENA_E00_CUDA_PROBE"]],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=float(os.environ["CATENA_E00_IDENTITY_TIMEOUT"]),
    check=False,
)
if identity_process.returncode != 0:
    raise RuntimeError(
        "native identity probe failed: "
        + (identity_process.stderr.strip() or identity_process.stdout.strip())
    )
identity = json.loads(identity_process.stdout.strip().splitlines()[-1])

payload = {
    "passed": False,
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "visible_device_count": torch.cuda.device_count(),
    "device_uuid": identity.get("uuid"),
    "pci_bus_id": identity.get("pci_bus_id"),
}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
torch.cuda.set_device(0)
properties = torch.cuda.get_device_properties(0)
payload.update(
    {
        "name": properties.name,
        "total_memory": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
)
if not payload["bf16_supported"]:
    raise RuntimeError("native BF16 is not supported")
size = 256
left = torch.ones((size, size), device="cuda", dtype=torch.bfloat16)
right = torch.ones((size, size), device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
started = time.perf_counter()
result = left @ right
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - started) * 1000.0
result_fp32 = result.float()
expected = float(size)
max_abs_error = float((result_fp32 - expected).abs().max().item())
finite = bool(torch.isfinite(result_fp32).all().item())
payload.update(
    {
        "matrix_size": size,
        "dtype": str(result.dtype),
        "finite": finite,
        "max_abs_error": max_abs_error,
        "elapsed_ms": elapsed_ms,
    }
)
payload["passed"] = finite and max_abs_error == 0.0 and result.dtype == torch.bfloat16
print(json.dumps(payload))
raise SystemExit(0 if payload["passed"] else 1)
""".strip()

TORCH_ENV_PROBE = r"""
import json
import os

import torch

devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "visible_index": index,
                "name": properties.name,
                "total_memory": properties.total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
payload = {
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "visible_device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "devices": devices,
    "torch_config": torch.__config__.show(),
}
print(json.dumps(payload))
""".strip()


class E00ConfigError(ValueError):
    """Raised when the E00 configuration is unsafe or incomplete."""


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


@dataclass
class AuditCheck:
    check_id: str
    category: str
    status: str
    required: bool
    summary: str
    expected: Any = None
    observed: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    blocked_by: list[str] = field(default_factory=list)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _version_tuple(value: str) -> tuple[int, ...]:
    base = value.split("+", 1)[0]
    numbers = re.findall(r"\d+", base)
    return tuple(int(number) for number in numbers[:4])


def _canonical_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _compare_versions(left: str, right: str) -> int:
    left_parts = list(_version_tuple(left))
    right_parts = list(_version_tuple(right))
    width = max(len(left_parts), len(right_parts))
    left_parts.extend([0] * (width - len(left_parts)))
    right_parts.extend([0] * (width - len(right_parts)))
    return (left_parts > right_parts) - (left_parts < right_parts)


def _version_satisfies(version: str, specification: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(version) in SpecifierSet(specification)
    except Exception:  # noqa: BLE001,S110 - retain the dependency-free fallback.
        pass

    for clause in (item.strip() for item in specification.split(",")):
        if not clause:
            continue
        match = re.fullmatch(r"(==|>=|<=|>|<)\s*([0-9][A-Za-z0-9.+-]*)", clause)
        if match is None:
            return False
        operator, target = match.groups()
        comparison = _compare_versions(version, target)
        if operator == "==" and version != target:
            return False
        if operator == ">=" and comparison < 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
    return True


def _resolve_repo_path(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise E00ConfigError(f"{label} escapes repository root: {value}") from exc
    return resolved


def _require_mapping(payload: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def validate_e00_config(config: Mapping[str, Any], root: str | Path | None = None) -> list[str]:
    errors: list[str] = []

    def reject_unknown(
        section_name: str, section: Mapping[str, Any], allowed: set[str]
    ) -> None:
        unknown_keys = sorted(set(section) - allowed)
        if unknown_keys:
            errors.append(
                f"unknown {section_name} keys: {', '.join(unknown_keys)}"
            )

    def finite_positive(value: object) -> TypeGuard[int | float]:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0
        )

    unknown = sorted(set(config) - TOP_LEVEL_KEYS)
    if unknown:
        errors.append(f"unknown top-level keys: {', '.join(unknown)}")
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if config.get("experiment") != "e00_audit":
        errors.append("experiment must be e00_audit")
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        errors.append("output_dir must be a non-empty string")

    checks = config.get("checks")
    if not isinstance(checks, list) or not all(isinstance(item, str) for item in checks):
        errors.append("checks must be a list of strings")
    else:
        duplicates = sorted({item for item in checks if checks.count(item) > 1})
        unsupported = sorted(set(checks) - SUPPORTED_CHECKS)
        missing = sorted(SUPPORTED_CHECKS - set(checks))
        if duplicates:
            errors.append(f"duplicate checks: {', '.join(duplicates)}")
        if unsupported:
            errors.append(f"unsupported checks: {', '.join(unsupported)}")
        if missing:
            errors.append(f"mandatory checks missing: {', '.join(missing)}")

    environment = _require_mapping(config, "environment", errors)
    reject_unknown("environment", environment, ENVIRONMENT_KEYS)
    missing_environment = sorted(ENVIRONMENT_KEYS - set(environment))
    if missing_environment:
        errors.append(f"environment keys missing: {', '.join(missing_environment)}")
    if environment.get("conda_name") != "catena":
        errors.append("environment.conda_name must be catena")
    if environment.get("python_major_minor") != "3.11":
        errors.append("environment.python_major_minor must be 3.11")
    for key in (
        "torch_version",
        "torch_cuda_runtime",
        "driver_version",
        "minimum_glibc",
    ):
        if not isinstance(environment.get(key), str) or not environment.get(key):
            errors.append(f"environment.{key} must be a non-empty string")
    required_packages = environment.get("required_packages")
    if not isinstance(required_packages, Mapping) or not required_packages:
        errors.append("environment.required_packages must be an object")
    elif not all(
        isinstance(name, str)
        and bool(name.strip())
        and isinstance(specification, str)
        and bool(specification.strip())
        for name, specification in required_packages.items()
    ):
        errors.append(
            "environment.required_packages keys and constraints must be non-empty strings"
        )
    else:
        normalized_required = {
            _canonical_package_name(str(name)): str(specification)
            for name, specification in required_packages.items()
        }
        expected_torch_constraint = f"=={environment.get('torch_version', '')}"
        if normalized_required.get("torch") != expected_torch_constraint:
            errors.append(
                "environment.required_packages.torch must exactly match "
                "environment.torch_version"
            )

    hardware = _require_mapping(config, "hardware", errors)
    reject_unknown("hardware", hardware, HARDWARE_KEYS)
    missing_hardware = sorted(HARDWARE_KEYS - set(hardware))
    if missing_hardware:
        errors.append(f"hardware keys missing: {', '.join(missing_hardware)}")
    selected = hardware.get("selected_physical_gpus")
    expected_visible = hardware.get("expected_visible_gpu_count")
    if not isinstance(selected, list) or not selected or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in selected
    ):
        errors.append("hardware.selected_physical_gpus must be a non-empty list of indices")
    elif len(set(selected)) != len(selected):
        errors.append("hardware.selected_physical_gpus contains duplicates")
    elif len(selected) != 4:
        errors.append("hardware.selected_physical_gpus must contain exactly four indices")
    if (
        not isinstance(expected_visible, int)
        or isinstance(expected_visible, bool)
        or expected_visible != 4
    ):
        errors.append("hardware.expected_visible_gpu_count must be exactly 4")
    elif isinstance(selected, list) and len(selected) != expected_visible:
        errors.append(
            "hardware.expected_visible_gpu_count must equal selected_physical_gpus length"
        )
    if not isinstance(hardware.get("expected_model"), str) or not hardware.get(
        "expected_model"
    ):
        errors.append("hardware.expected_model must be a non-empty string")
    minimum_memory = hardware.get("minimum_memory_mib")
    if (
        not isinstance(minimum_memory, int)
        or isinstance(minimum_memory, bool)
        or minimum_memory <= 0
    ):
        errors.append("hardware.minimum_memory_mib must be a positive integer")
    if not re.fullmatch(r"\d+\.\d+", str(hardware.get("compute_capability", ""))):
        errors.append("hardware.compute_capability must have major.minor form")
    if hardware.get("mig_mode") not in {"Enabled", "Disabled"}:
        errors.append("hardware.mig_mode must be Enabled or Disabled")

    toolchain = _require_mapping(config, "toolchain", errors)
    reject_unknown("toolchain", toolchain, TOOLCHAIN_KEYS)
    missing_toolchain = sorted(TOOLCHAIN_KEYS - set(toolchain))
    if missing_toolchain:
        errors.append(f"toolchain keys missing: {', '.join(missing_toolchain)}")
    if not re.fullmatch(r"\d+\.\d+", str(toolchain.get("cuda_toolkit_release", ""))):
        errors.append("toolchain.cuda_toolkit_release must have major.minor form")
    for key in ("compile_timeout_seconds", "lane_timeout_seconds"):
        value = toolchain.get(key)
        if not finite_positive(value) or float(value) > 3600:
            errors.append(f"toolchain.{key} must be finite and in (0, 3600]")

    storage = _require_mapping(config, "storage", errors)
    reject_unknown("storage", storage, STORAGE_KEYS)
    missing_storage = sorted(STORAGE_KEYS - set(storage))
    if missing_storage:
        errors.append(f"storage keys missing: {', '.join(missing_storage)}")
    if not isinstance(storage.get("path"), str):
        errors.append("storage.path must be a string")
    if not isinstance(storage.get("model_cache_path"), str):
        errors.append("storage.model_cache_path must be a string")
    probe_size = storage.get("probe_size_mib")
    repeats = storage.get("repeats")
    if (
        not isinstance(probe_size, int)
        or isinstance(probe_size, bool)
        or not 1 <= probe_size <= 2048
    ):
        errors.append("storage.probe_size_mib must be between 1 and 2048")
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or not 1 <= repeats <= 10
    ):
        errors.append("storage.repeats must be between 1 and 10")
    recommended_free = storage.get("recommended_free_gib")
    if (
        not isinstance(recommended_free, (int, float))
        or isinstance(recommended_free, bool)
        or not math.isfinite(float(recommended_free))
        or recommended_free < 0
    ):
        errors.append("storage.recommended_free_gib must be non-negative")

    repository = _require_mapping(config, "repository", errors)
    reject_unknown("repository", repository, REPOSITORY_KEYS)
    missing_repository = sorted(REPOSITORY_KEYS - set(repository))
    if missing_repository:
        errors.append(f"repository keys missing: {', '.join(missing_repository)}")
    for key in ("run_pytest", "run_config_audit", "run_mock_smoke"):
        if repository.get(key) is not True:
            errors.append(f"repository.{key} is a mandatory hard gate and must be true")

    if root is not None:
        root_path = Path(root).resolve()
        for path_value, label in (
            (output_dir, "output_dir"),
            (storage.get("path"), "storage.path"),
            (storage.get("model_cache_path"), "storage.model_cache_path"),
        ):
            if isinstance(path_value, str):
                try:
                    resolved = _resolve_repo_path(root_path, path_value, label)
                    relative = resolved.relative_to(root_path)
                    if not relative.parts:
                        errors.append(f"{label} must not be the repository root")
                    elif label == "output_dir" and relative.parts[0] != "artifacts":
                        errors.append("output_dir must be below artifacts/")
                    elif label != "output_dir" and relative.parts[0] not in {
                        "artifacts",
                        ".scratch",
                    }:
                        errors.append(
                            f"{label} must be below artifacts/ or .scratch/"
                        )
                except E00ConfigError as exc:
                    errors.append(str(exc))
        pyproject_path = root_path / "pyproject.toml"
        if pyproject_path.is_file() and isinstance(required_packages, Mapping):
            try:
                pyproject = tomllib.loads(
                    pyproject_path.read_text(encoding="utf-8")
                )
                project = pyproject.get("project", {})
                declared_requirements = list(project.get("dependencies", []))
                for group in project.get("optional-dependencies", {}).values():
                    declared_requirements.extend(group)
                declared_constraints: dict[str, str] = {}
                for requirement in declared_requirements:
                    requirement_text = str(requirement).strip()
                    match = re.match(
                        r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)",
                        requirement_text,
                    )
                    if match is not None:
                        declared_constraints[
                            _canonical_package_name(match.group(1))
                        ] = requirement_text[match.end() :].replace(" ", "")
                configured_names = {
                    _canonical_package_name(str(name))
                    for name in required_packages
                }
                missing_declared = sorted(
                    set(declared_constraints) - configured_names
                )
                if missing_declared:
                    errors.append(
                        "environment.required_packages omits pyproject dependencies: "
                        + ", ".join(missing_declared)
                    )
                configured_constraints = {
                    _canonical_package_name(str(name)): str(specification).replace(
                        " ", ""
                    )
                    for name, specification in required_packages.items()
                }
                constraint_mismatches = sorted(
                    name
                    for name, declared in declared_constraints.items()
                    if configured_constraints.get(name) not in {None, declared}
                )
                if constraint_mismatches:
                    errors.append(
                        "environment.required_packages constraints differ from "
                        "pyproject.toml: "
                        + ", ".join(constraint_mismatches)
                    )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                errors.append(f"could not validate pyproject dependencies: {exc}")
    return errors


def _parse_e00_config_text(text: str, root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise E00ConfigError(
                "E00 config must be JSON-compatible YAML when PyYAML is unavailable"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise E00ConfigError("E00 config root must be an object")
    errors = validate_e00_config(payload, root)
    if errors:
        raise E00ConfigError("; ".join(errors))
    return payload


def load_e00_config(path: str | Path, root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    config_path = _resolve_repo_path(root_path, str(path), "config path")
    text = config_path.read_text(encoding="utf-8")
    return _parse_e00_config_text(text, root_path)


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    started = time.perf_counter()
    command = [str(item) for item in argv]
    process: subprocess.Popen[str] | None = None

    def normalize_output(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return CommandResult(
            argv=command,
            returncode=process.returncode,
            stdout=normalize_output(stdout),
            stderr=normalize_output(stderr),
            elapsed_seconds=time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = normalize_output(exc.stdout)
        stderr = normalize_output(exc.stderr)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                final_stdout, final_stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                final_stdout, final_stderr = process.communicate()
            stdout = normalize_output(final_stdout) or stdout
            stderr = normalize_output(final_stderr) or stderr
        return CommandResult(
            argv=command,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=time.perf_counter() - started,
            timed_out=True,
            error=f"timeout after {timeout} seconds",
        )
    except OSError as exc:
        return CommandResult(
            argv=command,
            returncode=None,
            stdout="",
            stderr="",
            elapsed_seconds=time.perf_counter() - started,
            error=repr(exc),
        )


def _parse_nvidia_inventory(text: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            raise ValueError(f"unexpected nvidia-smi inventory row: {line}")
        index, name, uuid, driver, memory, pci_bus_id, capability, mig_mode = fields
        devices.append(
            {
                "physical_index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_mib": int(memory),
                "pci_bus_id": pci_bus_id,
                "compute_capability": capability,
                "mig_mode": mig_mode,
            }
        )
    return devices


def _normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"([0-9A-Fa-f]+):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])",
        value.strip(),
    )
    if match is None:
        return value.strip().lower()
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}.{function}"


def _probe_identity_matches(
    probe: Mapping[str, Any],
    inventory_device: Mapping[str, Any],
    *,
    uuid_key: str = "uuid",
) -> bool:
    capability = probe.get("compute_capability")
    if (
        isinstance(capability, Sequence)
        and not isinstance(capability, (str, bytes))
        and len(capability) == 2
    ):
        capability_text = f"{capability[0]}.{capability[1]}"
    else:
        capability_text = str(capability)
    return (
        probe.get(uuid_key) == inventory_device.get("uuid")
        and _normalize_pci_bus_id(str(probe.get("pci_bus_id", "")))
        == _normalize_pci_bus_id(str(inventory_device.get("pci_bus_id", "")))
        and probe.get("name") == inventory_device.get("name")
        and capability_text == str(inventory_device.get("compute_capability"))
        and probe.get("visible_device_count") == 1
    )


def _source_tree_hash(root: Path, relative_files: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative_text in sorted(set(relative_files)):
        candidate = root / relative_text
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        if candidate.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(candidate).encode("utf-8"))
            digest.update(b"\0")
            continue
        if not candidate.is_file():
            digest.update(b"missing\0")
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            digest.update(b"outside-repository-skipped\0")
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(resolved).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


class E00Auditor:
    def __init__(
        self,
        root: Path,
        config_path: Path,
        config: dict[str, Any],
        *,
        config_source_bytes: bytes | None = None,
    ) -> None:
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.config = config
        self.config_source_bytes = (
            config_source_bytes
            if config_source_bytes is not None
            else self.config_path.read_bytes()
        )
        snapshot_config = _parse_e00_config_text(
            self.config_source_bytes.decode("utf-8"), self.root
        )
        if snapshot_config != config:
            raise E00ConfigError(
                "config changed between parsing and audit initialization"
            )
        self.config_source_sha256 = hashlib.sha256(
            self.config_source_bytes
        ).hexdigest()
        self.resolved_config_text = (
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )
        runtime_paths = {
            "TMPDIR": self.root / ".scratch" / "tmp",
            "XDG_CACHE_HOME": self.root / ".scratch" / "xdg_cache",
            "XDG_CONFIG_HOME": self.root / ".scratch" / "xdg_config",
            "XDG_DATA_HOME": self.root / ".scratch" / "xdg_data",
            "CUDA_CACHE_PATH": self.root / ".scratch" / "cuda_cache",
            "MPLCONFIGDIR": self.root / ".scratch" / "matplotlib",
            "NUMBA_CACHE_DIR": self.root / ".scratch" / "numba",
            "HF_HOME": self.root / ".scratch" / "huggingface",
            "HF_DATASETS_CACHE": self.root / ".scratch" / "huggingface" / "datasets",
            "TORCH_EXTENSIONS_DIR": self.root / ".scratch" / "torch_extensions",
            "TRITON_CACHE_DIR": self.root / ".scratch" / "triton",
            "TORCHINDUCTOR_CACHE_DIR": self.root / ".scratch" / "torchinductor",
            "WANDB_DIR": self.root / ".scratch" / "wandb",
            "WANDB_CACHE_DIR": self.root / ".scratch" / "wandb_cache",
            "WANDB_CONFIG_DIR": self.root / ".scratch" / "wandb_config",
            "WANDB_DATA_DIR": self.root / ".scratch" / "wandb_data",
            "PIP_CACHE_DIR": self.root / ".scratch" / "pip_cache",
            "PYTHONPYCACHEPREFIX": self.root / ".scratch" / "pycache",
        }
        for name, path in runtime_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        os.environ["PYTHONPATH"] = str(self.root / "src")
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output_base = _resolve_repo_path(
            self.root, str(config["output_dir"]), "output_dir"
        )
        self.run_dir = self.output_base / "runs" / self.run_id
        self.raw_dir = self.run_dir / "raw"
        self.build_dir = self.run_dir / "build"
        self.storage_artifact_dir = self.run_dir / "storage"
        self.repository_dir = self.run_dir / "repository"
        for directory in (
            self.raw_dir,
            self.build_dir,
            self.storage_artifact_dir,
            self.repository_dir,
        ):
            directory.mkdir(parents=True, exist_ok=False)
        self.checks: list[AuditCheck] = []
        self.host_gpus: list[dict[str, Any]] = []
        self.selected_gpus: list[dict[str, Any]] = []
        self.cuda_lane_results: list[dict[str, Any]] = []
        self.torch_lane_results: list[dict[str, Any]] = []
        self.storage_results: dict[str, Any] = {}
        self.installed_packages: dict[str, str] = {}
        self.git: dict[str, Any] = {}
        self.cuda_probe_binary: Path | None = None
        self.phase_errors: list[dict[str, str]] = []
        self.command_timeout = 60.0
        _atomic_write_text(
            self.run_dir / "resolved_config.yaml",
            self.resolved_config_text,
        )

    def add_check(
        self,
        check_id: str,
        category: str,
        status: str,
        summary: str,
        *,
        required: bool = True,
        expected: Any = None,
        observed: Any = None,
        details: Mapping[str, Any] | None = None,
        blocked_by: Sequence[str] = (),
    ) -> None:
        if any(check.check_id == check_id for check in self.checks):
            raise RuntimeError(f"duplicate E00 check id: {check_id}")
        self.checks.append(
            AuditCheck(
                check_id=check_id,
                category=category,
                status=status,
                required=required,
                summary=summary,
                expected=expected,
                observed=observed,
                details=dict(details or {}),
                blocked_by=list(blocked_by),
            )
        )

    def write_raw(self, name: str, result: CommandResult | str) -> None:
        path = self.raw_dir / name
        if isinstance(result, str):
            _atomic_write_text(path, result)
            return
        payload = {
            "argv": result.argv,
            "returncode": result.returncode,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "error": result.error,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        _atomic_write_json(path, payload)

    def check(self, check_id: str) -> AuditCheck | None:
        return next(
            (check for check in self.checks if check.check_id == check_id),
            None,
        )

    def command(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        return run_command(
            argv,
            cwd=self.root,
            timeout=timeout or self.command_timeout,
            env=env,
        )

    def audit_identity_and_packages(self) -> None:
        environment = self.config["environment"]
        expected_conda = str(environment["conda_name"])
        observed_conda = os.getenv("CONDA_DEFAULT_ENV")
        conda_prefix = os.getenv("CONDA_PREFIX")
        executable_in_prefix = False
        if conda_prefix:
            try:
                Path(sys.executable).resolve().relative_to(Path(conda_prefix).resolve())
                executable_in_prefix = True
            except ValueError:
                executable_in_prefix = False
        prefix_name_matches = bool(conda_prefix) and (
            Path(str(conda_prefix)).resolve().name == expected_conda
        )
        module_path = Path(__file__).resolve()
        try:
            module_path.relative_to(self.root / "src")
            module_from_repository = True
        except ValueError:
            module_from_repository = False
        conda_passed = (
            observed_conda == expected_conda
            and bool(conda_prefix)
            and executable_in_prefix
            and prefix_name_matches
            and module_from_repository
        )
        self.add_check(
            "conda_environment",
            "environment",
            PASS if conda_passed else FAIL,
            (
                f"Conda environment is {expected_conda}"
                if conda_passed
                else f"expected Conda {expected_conda}, observed {observed_conda!r}"
            ),
            expected=expected_conda,
            observed={
                "name": observed_conda,
                "prefix": conda_prefix,
                "prefix_name_matches": prefix_name_matches,
                "executable": sys.executable,
                "executable_in_prefix": executable_in_prefix,
                "module_path": str(module_path),
                "module_from_repository": module_from_repository,
            },
        )

        expected_python = str(environment["python_major_minor"])
        observed_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        python_passed = observed_python == expected_python
        self.add_check(
            "python_runtime",
            "environment",
            PASS if python_passed else FAIL,
            f"Python {platform.python_version()}",
            expected=expected_python,
            observed=platform.python_version(),
        )

        libc_name, libc_version = platform.libc_ver()
        minimum_glibc = str(environment["minimum_glibc"])
        glibc_passed = libc_name.lower() == "glibc" and _compare_versions(
            libc_version, minimum_glibc
        ) >= 0
        self.add_check(
            "glibc_runtime",
            "environment",
            PASS if glibc_passed else FAIL,
            f"{libc_name or 'unknown'} {libc_version or 'unknown'}",
            expected=f"glibc>={minimum_glibc}",
            observed=f"{libc_name} {libc_version}",
        )

        installed: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata["Name"]
            if name:
                installed[_canonical_package_name(name)] = distribution.version
        self.installed_packages = installed
        snapshot = "\n".join(
            f"{name}=={version}" for name, version in sorted(installed.items())
        )
        _atomic_write_text(self.run_dir / "package_snapshot.txt", snapshot + "\n")

        pip_freeze = self.command(
            [sys.executable, "-m", "pip", "freeze", "--all"], timeout=120
        )
        pip_check = self.command(
            [sys.executable, "-m", "pip", "check"], timeout=120
        )
        conda_explicit = self.command(
            ["conda", "list", "--explicit", "--prefix", str(conda_prefix or "")],
            timeout=120,
        )
        self.write_raw("pip_freeze_all.json", pip_freeze)
        self.write_raw("pip_check.json", pip_check)
        self.write_raw("conda_list_explicit.json", conda_explicit)
        _atomic_write_text(
            self.run_dir / "pip-freeze-all.txt",
            pip_freeze.stdout if pip_freeze.passed else "",
        )
        _atomic_write_text(
            self.run_dir / "conda-explicit.txt",
            conda_explicit.stdout if conda_explicit.passed else "",
        )

        required_packages: Mapping[str, str] = environment["required_packages"]
        package_results: list[dict[str, Any]] = []
        missing_or_wrong: list[str] = []
        for name, specification in required_packages.items():
            observed = installed.get(_canonical_package_name(name))
            passed = observed is not None and _version_satisfies(observed, str(specification))
            package_results.append(
                {
                    "name": name,
                    "expected": specification,
                    "observed": observed,
                    "passed": passed,
                }
            )
            if not passed:
                missing_or_wrong.append(
                    f"{name} ({observed or 'missing'}; expected {specification})"
                )
        snapshots_passed = pip_freeze.passed and pip_check.passed and conda_explicit.passed
        if not snapshots_passed:
            missing_or_wrong.append("environment snapshot or dependency consistency failed")
        self.add_check(
            "python_packages",
            "environment",
            PASS if not missing_or_wrong else FAIL,
            (
                "E00 Python packages satisfy the declared constraints"
                if not missing_or_wrong
                else "missing or incompatible packages: " + ", ".join(missing_or_wrong)
            ),
            expected=dict(required_packages),
            observed={
                "packages": package_results,
                "pip_freeze_captured": pip_freeze.passed,
                "pip_check_passed": pip_check.passed,
                "conda_explicit_captured": conda_explicit.passed,
            },
        )

    def audit_host_gpu_inventory(self) -> None:
        query = [
            "nvidia-smi",
            (
                "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id,"
                "compute_cap,mig.mode.current"
            ),
            "--format=csv,noheader,nounits",
        ]
        inventory = self.command(query)
        self.write_raw("nvidia_inventory.json", inventory)
        if not inventory.passed:
            self.add_check(
                "host_gpu_inventory",
                "hardware",
                FAIL,
                "nvidia-smi inventory failed",
                observed=asdict(inventory),
            )
            return
        try:
            self.host_gpus = _parse_nvidia_inventory(inventory.stdout)
        except Exception as exc:  # noqa: BLE001 - convert malformed inventory to a check.
            self.add_check(
                "host_gpu_inventory",
                "hardware",
                ERROR,
                f"could not parse nvidia-smi inventory: {exc}",
            )
            return

        hardware = self.config["hardware"]
        selected_indices = list(hardware["selected_physical_gpus"])
        by_index = {device["physical_index"]: device for device in self.host_gpus}
        missing = [index for index in selected_indices if index not in by_index]
        self.selected_gpus = [by_index[index] for index in selected_indices if index in by_index]
        uuids = [device["uuid"] for device in self.selected_gpus]
        policy_mismatches: list[dict[str, Any]] = []
        expected_model = str(hardware["expected_model"])
        minimum_memory_mib = int(hardware["minimum_memory_mib"])
        expected_capability = str(hardware["compute_capability"])
        expected_mig_mode = str(hardware["mig_mode"])
        for device in self.selected_gpus:
            mismatches: dict[str, Any] = {}
            if device["name"] != expected_model:
                mismatches["name"] = {
                    "expected": expected_model,
                    "observed": device["name"],
                }
            if int(device["memory_mib"]) < minimum_memory_mib:
                mismatches["memory_mib"] = {
                    "minimum": minimum_memory_mib,
                    "observed": device["memory_mib"],
                }
            if device["compute_capability"] != expected_capability:
                mismatches["compute_capability"] = {
                    "expected": expected_capability,
                    "observed": device["compute_capability"],
                }
            if device["mig_mode"] != expected_mig_mode:
                mismatches["mig_mode"] = {
                    "expected": expected_mig_mode,
                    "observed": device["mig_mode"],
                }
            if mismatches:
                policy_mismatches.append(
                    {
                        "physical_index": device["physical_index"],
                        "mismatches": mismatches,
                    }
                )
        inventory_passed = (
            not missing
            and len(self.selected_gpus) == int(hardware["expected_visible_gpu_count"])
            and len(set(uuids)) == len(selected_indices)
            and not policy_mismatches
        )
        self.add_check(
            "host_gpu_inventory",
            "hardware",
            PASS if inventory_passed else FAIL,
            (
                f"selected {len(self.selected_gpus)} of {len(self.host_gpus)} host GPUs"
                if inventory_passed
                else (
                    "selected GPU inventory invalid; "
                    f"missing={missing}, policy_mismatches={len(policy_mismatches)}"
                )
            ),
            expected={
                "selected_physical_gpus": selected_indices,
                "model": expected_model,
                "minimum_memory_mib": minimum_memory_mib,
                "compute_capability": expected_capability,
                "mig_mode": expected_mig_mode,
            },
            observed={
                "host_gpus": self.host_gpus,
                "policy_mismatches": policy_mismatches,
            },
        )
        if len(self.host_gpus) != len(selected_indices):
            self.add_check(
                "extra_host_gpus",
                "hardware",
                WARN,
                (
                    f"host exposes {len(self.host_gpus)} GPUs; experiment allocation is pinned "
                    f"to {selected_indices}"
                ),
                required=False,
                observed=len(self.host_gpus),
            )

        expected_driver = str(self.config["environment"]["driver_version"])
        driver_versions = sorted(
            {device["driver_version"] for device in self.selected_gpus}
        )
        driver_passed = bool(driver_versions) and driver_versions == [expected_driver]
        self.add_check(
            "nvidia_driver",
            "hardware",
            PASS if driver_passed else FAIL,
            f"NVIDIA driver versions: {', '.join(driver_versions) or 'unavailable'}",
            expected=expected_driver,
            observed=driver_versions,
        )

        for command_name, argv in (
            ("nvidia_topology.json", ["nvidia-smi", "topo", "-m"]),
            ("nvidia_p2p_read.json", ["nvidia-smi", "topo", "-p2p", "r"]),
            ("nvidia_p2p_write.json", ["nvidia-smi", "topo", "-p2p", "w"]),
        ):
            result = self.command(argv)
            self.write_raw(command_name, result)
            self.add_check(
                command_name.removesuffix(".json"),
                "hardware",
                PASS if result.passed else WARN,
                f"{'captured' if result.passed else 'failed to capture'} {' '.join(argv[1:])}",
                required=False,
                observed={"returncode": result.returncode, "error": result.error},
            )

    def audit_cuda_toolchain_and_lanes(self) -> None:
        toolchain = self.config["toolchain"]
        nvcc = self.command(["nvcc", "--version"])
        gcc = self.command(["gcc", "--version"])
        gxx = self.command(["g++", "--version"])
        self.write_raw("nvcc.json", nvcc)
        self.write_raw("gcc.json", gcc)
        self.write_raw("gxx.json", gxx)

        expected_cuda = str(toolchain["cuda_toolkit_release"])
        match = re.search(r"release\s+([0-9]+\.[0-9]+)", nvcc.stdout)
        observed_cuda = match.group(1) if match else None
        commands_passed = nvcc.passed and gcc.passed and gxx.passed
        version_passed = observed_cuda == expected_cuda
        self.add_check(
            "cuda_toolchain",
            "toolchain",
            PASS if commands_passed and version_passed else FAIL,
            (
                f"nvcc {observed_cuda}; gcc/g++ available"
                if commands_passed
                else "CUDA/C++ compiler command failed"
            ),
            expected={"cuda_toolkit_release": expected_cuda},
            observed={
                "cuda_toolkit_release": observed_cuda,
                "nvcc_returncode": nvcc.returncode,
                "gcc_returncode": gcc.returncode,
                "gxx_returncode": gxx.returncode,
            },
        )
        inventory_check = self.check("host_gpu_inventory")
        expected_count = int(self.config["hardware"]["expected_visible_gpu_count"])
        inventory_ready = (
            inventory_check is not None
            and inventory_check.status == PASS
            and len(self.selected_gpus) == expected_count
        )
        if not commands_passed or not version_passed or not inventory_ready:
            self.add_check(
                "cuda_bf16_lanes",
                "hardware",
                BLOCKED,
                "CUDA BF16 lane probe blocked by inventory or toolchain failure",
                blocked_by=["host_gpu_inventory", "cuda_toolchain"],
            )
            return

        source = self.root / "scripts" / "e00_cuda_bf16_probe.cu"
        binary = self.build_dir / "e00_cuda_bf16_probe"
        if not source.is_file():
            self.add_check(
                "cuda_probe_compile",
                "toolchain",
                FAIL,
                "CUDA BF16 probe source is missing",
                observed=source.relative_to(self.root).as_posix(),
            )
            self.add_check(
                "cuda_bf16_lanes",
                "hardware",
                BLOCKED,
                "CUDA BF16 lane probe blocked because its source is missing",
                blocked_by=["cuda_probe_compile"],
            )
            return
        capabilities = {
            str(device["compute_capability"]) for device in self.selected_gpus
        }
        if len(capabilities) != 1:
            self.add_check(
                "cuda_probe_compile",
                "toolchain",
                FAIL,
                f"selected GPUs have heterogeneous compute capabilities: {sorted(capabilities)}",
            )
            self.add_check(
                "cuda_bf16_lanes",
                "hardware",
                BLOCKED,
                "CUDA BF16 lane probe blocked by heterogeneous GPU architectures",
                blocked_by=["cuda_probe_compile"],
            )
            return
        capability = next(iter(capabilities))
        architecture = capability.replace(".", "")
        compile_result = self.command(
            [
                "nvcc",
                "-std=c++17",
                "-O2",
                f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
                str(source),
                "-lcublas",
                "-lcuda",
                "-o",
                str(binary),
            ],
            timeout=float(toolchain["compile_timeout_seconds"]),
        )
        self.write_raw("cuda_probe_build.json", compile_result)
        self.add_check(
            "cuda_probe_compile",
            "toolchain",
            PASS if compile_result.passed else FAIL,
            (
                "CUDA BF16 probe compiled and linked"
                if compile_result.passed
                else "CUDA BF16 probe compilation failed"
            ),
            observed=asdict(compile_result),
        )
        if not compile_result.passed:
            self.add_check(
                "cuda_bf16_lanes",
                "hardware",
                BLOCKED,
                "CUDA BF16 lane probe blocked by compilation failure",
                blocked_by=["cuda_probe_compile"],
            )
            return
        self.cuda_probe_binary = binary

        timeout = float(toolchain["lane_timeout_seconds"])

        def launch(device: dict[str, Any]) -> tuple[dict[str, Any], CommandResult]:
            lane_environment = os.environ.copy()
            lane_environment["CUDA_VISIBLE_DEVICES"] = device["uuid"]
            result = self.command([str(binary)], timeout=timeout, env=lane_environment)
            return device, result

        lane_payloads: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.selected_gpus)
        ) as executor:
            futures = [executor.submit(launch, device) for device in self.selected_gpus]
            for future in concurrent.futures.as_completed(futures):
                device, result = future.result()
                lane_name = f"cuda_lane_gpu{device['physical_index']}.json"
                self.write_raw(lane_name, result)
                payload: dict[str, Any] = {
                    "physical_index": device["physical_index"],
                    "uuid": device["uuid"],
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "error": result.error,
                    "passed": False,
                }
                if result.stdout.strip():
                    try:
                        probe = json.loads(result.stdout.strip().splitlines()[-1])
                        payload["probe"] = probe
                        identity_matches = _probe_identity_matches(probe, device)
                        payload["identity_matches_inventory"] = identity_matches
                        payload["passed"] = bool(
                            result.passed
                            and probe.get("passed")
                            and probe.get("sm_target_kernel_executed")
                            and identity_matches
                        )
                    except json.JSONDecodeError as exc:
                        payload["parse_error"] = repr(exc)
                lane_payloads.append(payload)
        lane_payloads.sort(key=lambda item: item["physical_index"])
        self.cuda_lane_results = lane_payloads
        lanes_passed = (
            len(lane_payloads) == len(self.selected_gpus)
            and all(payload["passed"] for payload in lane_payloads)
        )
        self.add_check(
            "cuda_bf16_lanes",
            "hardware",
            PASS if lanes_passed else FAIL,
            (
                "all selected GPUs passed concurrent BF16-input cuBLAS GEMM, "
                "sm-target kernel, and device-identity checks"
                if lanes_passed
                else "one or more CUDA BF16 or device-identity lanes failed"
            ),
            expected={
                "lane_count": expected_count,
                "visible_per_lane": 1,
                "identity_matches_inventory": True,
                "sm_target_kernel_executed": True,
            },
            observed=lane_payloads,
        )

    def audit_pytorch(self) -> None:
        environment = self.config["environment"]
        hardware = self.config["hardware"]
        expected_count = int(hardware["expected_visible_gpu_count"])
        inventory_check = self.check("host_gpu_inventory")
        inventory_ready = (
            inventory_check is not None
            and inventory_check.status == PASS
            and len(self.selected_gpus) == expected_count
        )
        if not inventory_ready:
            self.add_check(
                "pytorch_cuda",
                "pytorch",
                BLOCKED,
                "PyTorch/CUDA gate blocked by invalid selected GPU inventory",
                blocked_by=["host_gpu_inventory"],
            )
            self.add_check(
                "pytorch_bf16_lanes",
                "pytorch",
                BLOCKED,
                "PyTorch BF16 lanes blocked by invalid selected GPU inventory",
                blocked_by=["host_gpu_inventory", "pytorch_cuda"],
            )
            return

        expected_torch = str(environment["torch_version"])
        expected_runtime = str(environment["torch_cuda_runtime"])
        selected_uuids = [device["uuid"] for device in self.selected_gpus]
        command_environment = os.environ.copy()
        command_environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected_uuids)
        command_environment["PYTHONPATH"] = str(self.root / "src")
        environment_result = self.command(
            [sys.executable, "-c", TORCH_ENV_PROBE],
            timeout=float(self.config["toolchain"]["lane_timeout_seconds"]),
            env=command_environment,
        )
        self.write_raw("pytorch_environment_command.json", environment_result)
        observed: dict[str, Any] = {}
        if environment_result.stdout.strip():
            try:
                observed = json.loads(
                    environment_result.stdout.strip().splitlines()[-1]
                )
            except json.JSONDecodeError as exc:
                observed = {"parse_error": repr(exc)}
        _atomic_write_json(self.raw_dir / "pytorch_environment.json", observed)
        observed_devices = observed.get("devices")
        device_metadata_matches = (
            isinstance(observed_devices, list)
            and len(observed_devices) == expected_count
            and all(
                observed_device.get("name") == selected_device["name"]
                and observed_device.get("compute_capability")
                == [
                    int(part)
                    for part in selected_device["compute_capability"].split(".", 1)
                ]
                and observed_device.get("bf16_supported") is True
                for observed_device, selected_device in zip(
                    observed_devices, self.selected_gpus
                )
            )
        )
        pytorch_passed = (
            environment_result.passed
            and observed.get("torch") == expected_torch
            and observed.get("torch_cuda_runtime") == expected_runtime
            and observed.get("cuda_available") is True
            and observed.get("visible_device_count") == expected_count
            and observed.get("cuda_visible_devices") == ",".join(selected_uuids)
            and device_metadata_matches
        )
        self.add_check(
            "pytorch_cuda",
            "pytorch",
            PASS if pytorch_passed else FAIL,
            (
                f"PyTorch {observed.get('torch')}, CUDA "
                f"{observed.get('torch_cuda_runtime')}, "
                f"{observed.get('visible_device_count')} selected GPUs"
                if environment_result.passed
                else (
                    "PyTorch environment probe failed: "
                    + (
                        environment_result.stderr.strip().splitlines()[-1]
                        if environment_result.stderr.strip()
                        else environment_result.error or "no diagnostic"
                    )
                )
            ),
            expected={
                "torch": expected_torch,
                "cuda": expected_runtime,
                "visible_device_count": expected_count,
                "cuda_visible_devices": selected_uuids,
                "device_metadata_matches_inventory": True,
            },
            observed={
                "probe": observed,
                "device_metadata_matches_inventory": device_metadata_matches,
                "command": {
                    "returncode": environment_result.returncode,
                    "timed_out": environment_result.timed_out,
                    "error": environment_result.error,
                },
            },
        )
        cuda_lanes = self.check("cuda_bf16_lanes")
        if (
            not pytorch_passed
            or cuda_lanes is None
            or cuda_lanes.status != PASS
            or self.cuda_probe_binary is None
        ):
            blocked_by: list[str] = []
            for check_id in ("pytorch_cuda", "cuda_bf16_lanes"):
                prerequisite = self.check(check_id)
                if prerequisite is None or prerequisite.status != PASS:
                    blocked_by.append(check_id)
            self.add_check(
                "pytorch_bf16_lanes",
                "pytorch",
                BLOCKED,
                "PyTorch BF16 lanes blocked by the PyTorch/CUDA identity gates",
                blocked_by=blocked_by,
            )
            return

        timeout = float(self.config["toolchain"]["lane_timeout_seconds"])

        def launch(device: dict[str, Any]) -> tuple[dict[str, Any], CommandResult]:
            lane_environment = os.environ.copy()
            lane_environment["CUDA_VISIBLE_DEVICES"] = device["uuid"]
            lane_environment["PYTHONPATH"] = str(self.root / "src")
            lane_environment["CATENA_E00_CUDA_PROBE"] = str(
                self.cuda_probe_binary
            )
            lane_environment["CATENA_E00_IDENTITY_TIMEOUT"] = str(
                min(30.0, timeout)
            )
            result = self.command(
                [sys.executable, "-c", TORCH_LANE_PROBE],
                timeout=timeout,
                env=lane_environment,
            )
            return device, result

        lane_payloads: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.selected_gpus)
        ) as executor:
            futures = [executor.submit(launch, device) for device in self.selected_gpus]
            for future in concurrent.futures.as_completed(futures):
                device, result = future.result()
                self.write_raw(
                    f"pytorch_lane_gpu{device['physical_index']}.json", result
                )
                payload: dict[str, Any] = {
                    "physical_index": device["physical_index"],
                    "uuid": device["uuid"],
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "error": result.error,
                    "passed": False,
                }
                if result.stdout.strip():
                    try:
                        probe = json.loads(result.stdout.strip().splitlines()[-1])
                        payload["probe"] = probe
                        identity_matches = _probe_identity_matches(
                            probe,
                            device,
                            uuid_key="device_uuid",
                        )
                        payload["identity_matches_inventory"] = identity_matches
                        payload["passed"] = bool(
                            result.passed and probe.get("passed") and identity_matches
                        )
                    except json.JSONDecodeError as exc:
                        payload["parse_error"] = repr(exc)
                lane_payloads.append(payload)
        lane_payloads.sort(key=lambda item: item["physical_index"])
        self.torch_lane_results = lane_payloads
        lanes_passed = (
            len(lane_payloads) == len(self.selected_gpus)
            and all(payload["passed"] for payload in lane_payloads)
        )
        self.add_check(
            "pytorch_bf16_lanes",
            "pytorch",
            PASS if lanes_passed else FAIL,
            (
                "all selected GPUs passed concurrent PyTorch BF16 matmul "
                "and device-identity checks"
                if lanes_passed
                else "one or more PyTorch BF16 or device-identity lanes failed"
            ),
            expected={
                "lane_count": expected_count,
                "visible_per_lane": 1,
                "identity_matches_inventory": True,
            },
            observed=lane_payloads,
        )

    def audit_storage(self) -> None:
        storage = self.config["storage"]
        targets = {
            "state_cache": _resolve_repo_path(
                self.root, str(storage["path"]), "storage.path"
            ),
            "model_cache": _resolve_repo_path(
                self.root,
                str(storage["model_cache_path"]),
                "storage.model_cache_path",
            ),
        }
        capacity_rows: list[dict[str, Any]] = []
        for label, target in targets.items():
            target.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(target)
            capacity_rows.append(
                {
                    "label": label,
                    "path": target.relative_to(self.root).as_posix(),
                    "filesystem_device": os.stat(target).st_dev,
                    "free_gib": usage.free / (1024**3),
                    "total_gib": usage.total / (1024**3),
                }
            )
        free_gib = min(row["free_gib"] for row in capacity_rows)
        recommended_free = float(storage["recommended_free_gib"])
        if free_gib < recommended_free:
            self.add_check(
                "storage_capacity",
                "storage",
                WARN,
                (
                    f"{free_gib:.2f} GiB free is below the recommended "
                    f"{recommended_free:.2f} GiB"
                ),
                required=False,
                expected={"recommended_free_gib": recommended_free},
                observed=capacity_rows,
            )
        else:
            self.add_check(
                "storage_capacity",
                "storage",
                PASS,
                f"{free_gib:.2f} GiB free",
                required=False,
                expected={"recommended_free_gib": recommended_free},
                observed=capacity_rows,
            )

        size_bytes = int(storage["probe_size_mib"]) * 1024 * 1024
        repeats = int(storage["repeats"])
        block = hashlib.sha256(b"CATENA-E00-STORAGE-PROBE").digest() * (1024 * 1024 // 32)
        target_results: dict[str, dict[str, Any]] = {}
        all_passed = True
        aggregate_write_speeds: list[float] = []
        aggregate_read_speeds: list[float] = []
        for label, target in targets.items():
            runs: list[dict[str, Any]] = []
            for index in range(repeats):
                probe = target / f"probe-{self.run_id}-{index}.bin"
                write_digest = hashlib.sha256()
                read_digest = hashlib.sha256()
                written = 0
                read = 0
                try:
                    started = time.perf_counter()
                    with probe.open("wb", buffering=0) as stream:
                        while written < size_bytes:
                            chunk = block[: min(len(block), size_bytes - written)]
                            count = stream.write(chunk)
                            if count != len(chunk):
                                raise OSError(
                                    f"short write: {count} of {len(chunk)}"
                                )
                            write_digest.update(chunk)
                            written += count
                        os.fsync(stream.fileno())
                    write_seconds = time.perf_counter() - started

                    if hasattr(os, "posix_fadvise") and hasattr(
                        os, "POSIX_FADV_DONTNEED"
                    ):
                        with probe.open("rb", buffering=0) as stream:
                            os.posix_fadvise(
                                stream.fileno(),
                                0,
                                0,
                                os.POSIX_FADV_DONTNEED,
                            )

                    started = time.perf_counter()
                    with probe.open("rb", buffering=0) as stream:
                        while chunk := stream.read(len(block)):
                            read_digest.update(chunk)
                            read += len(chunk)
                    read_seconds = time.perf_counter() - started
                    passed = (
                        written == size_bytes
                        and read == size_bytes
                        and write_digest.hexdigest() == read_digest.hexdigest()
                    )
                    write_mib_s = size_bytes / (1024**2) / write_seconds
                    read_mib_s = size_bytes / (1024**2) / read_seconds
                    run = {
                        "iteration": index,
                        "bytes": size_bytes,
                        "write_seconds": write_seconds,
                        "read_seconds": read_seconds,
                        "write_mib_s": write_mib_s,
                        "read_mib_s": read_mib_s,
                        "sha256": write_digest.hexdigest(),
                        "passed": passed,
                        "cold_read_not_guaranteed": True,
                    }
                    aggregate_write_speeds.append(write_mib_s)
                    aggregate_read_speeds.append(read_mib_s)
                except Exception as exc:  # noqa: BLE001 - isolate each storage probe.
                    passed = False
                    run = {
                        "iteration": index,
                        "bytes_written": written,
                        "bytes_read": read,
                        "passed": False,
                        "error": repr(exc),
                    }
                finally:
                    probe.unlink(missing_ok=True)
                all_passed = all_passed and passed
                runs.append(run)
            target_results[label] = {
                "path": target.relative_to(self.root).as_posix(),
                "filesystem_device": os.stat(target).st_dev,
                "runs": runs,
                "median_write_mib_s": statistics.median(
                    [
                        run["write_mib_s"]
                        for run in runs
                        if "write_mib_s" in run
                    ]
                )
                if any("write_mib_s" in run for run in runs)
                else None,
                "median_read_mib_s": statistics.median(
                    [
                        run["read_mib_s"]
                        for run in runs
                        if "read_mib_s" in run
                    ]
                )
                if any("read_mib_s" in run for run in runs)
                else None,
            }

        self.storage_results = {
            "targets": target_results,
            "capacity": capacity_rows,
            "free_gib_before": free_gib,
            "recommended_free_gib": recommended_free,
            "probe_size_mib": int(storage["probe_size_mib"]),
            "repeats": repeats,
            "median_write_mib_s": statistics.median(aggregate_write_speeds)
            if aggregate_write_speeds
            else None,
            "median_read_mib_s": statistics.median(aggregate_read_speeds)
            if aggregate_read_speeds
            else None,
        }
        _atomic_write_json(self.storage_artifact_dir / "storage_results.json", self.storage_results)
        self.add_check(
            "state_cache_storage",
            "storage",
            PASS if all_passed else FAIL,
            (
                f"state/model caches passed {repeats} integrity probes each"
                if all_passed
                else "state-cache or model-cache storage integrity probe failed"
            ),
            expected={
                "targets": sorted(targets),
                "runs_per_target": repeats,
                "bytes_per_run": size_bytes,
                "sha256_match": True,
            },
            observed=self.storage_results,
        )

    def audit_repository(self) -> None:
        git_commit = self.command(["git", "rev-parse", "HEAD"])
        git_status = self.command(["git", "status", "--short"])
        git_files = self.command(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ]
        )
        self.write_raw("git_commit.json", git_commit)
        self.write_raw("git_status.json", git_status)
        self.write_raw("git_files.json", git_files)
        relative_files = (
            [item for item in git_files.stdout.split("\0") if item]
            if git_files.passed
            else []
        )
        self.git = {
            "commit": git_commit.stdout.strip() if git_commit.passed else None,
            "dirty": bool(git_status.stdout.strip()) if git_status.passed else None,
            "status": git_status.stdout.splitlines() if git_status.passed else [],
            "source_tree_file_count": len(relative_files),
            "source_tree_sha256": (
                _source_tree_hash(self.root, relative_files)
                if git_files.passed
                else None
            ),
        }
        if git_commit.passed and git_status.passed and git_files.passed:
            self.add_check(
                "git_snapshot",
                "repository",
                PASS,
                f"git commit recorded; dirty={self.git['dirty']}",
                observed=self.git,
            )
            if self.git["dirty"]:
                self.add_check(
                    "git_dirty",
                    "repository",
                    WARN,
                    "repository has tracked or untracked changes; snapshot hash was recorded",
                    required=False,
                    observed=self.git["status"],
                )
        else:
            self.add_check(
                "git_snapshot",
                "repository",
                FAIL,
                "could not capture git commit/status",
                observed={
                    "commit": asdict(git_commit),
                    "status": asdict(git_status),
                    "files": asdict(git_files),
                },
            )

        compile_result = self.command(
            [sys.executable, "-m", "compileall", "-q", "src"], timeout=120
        )
        self.write_raw("python_compileall.json", compile_result)
        self.add_check(
            "python_compileall",
            "repository",
            PASS if compile_result.passed else FAIL,
            (
                "Python sources compile"
                if compile_result.passed
                else "Python source compilation failed"
            ),
            observed=asdict(compile_result),
        )

        shell_results: list[dict[str, Any]] = []
        for script in sorted((self.root / "scripts").glob("*.sh")):
            result = self.command(["bash", "-n", str(script)], timeout=30)
            shell_results.append(
                {
                    "path": script.relative_to(self.root).as_posix(),
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "passed": result.passed,
                }
            )
        _atomic_write_json(self.repository_dir / "shell_syntax.json", shell_results)
        shell_passed = bool(shell_results) and all(item["passed"] for item in shell_results)
        self.add_check(
            "shell_syntax",
            "repository",
            PASS if shell_passed else FAIL,
            (
                f"{len(shell_results)} shell scripts passed syntax checks"
                if shell_passed
                else "one or more shell scripts failed syntax checks"
            ),
            observed=shell_results,
        )

        repository_config = self.config["repository"]
        command_environment = os.environ.copy()
        command_environment["PYTHONPATH"] = str(self.root / "src")
        pytest_version = self.installed_packages.get("pytest")
        if repository_config["run_pytest"] and pytest_version:
            pytest_result = self.command(
                [sys.executable, "-m", "pytest", "-q"],
                timeout=300,
                env=command_environment,
            )
            self.write_raw("pytest.json", pytest_result)
            self.add_check(
                "repository_pytest",
                "repository",
                PASS if pytest_result.passed else FAIL,
                (
                    "repository pytest suite passed"
                    if pytest_result.passed
                    else "repository pytest suite failed"
                ),
                observed=asdict(pytest_result),
            )
        elif repository_config["run_pytest"]:
            self.add_check(
                "repository_pytest",
                "repository",
                BLOCKED,
                "pytest suite blocked because pytest is not installed",
                blocked_by=["python_packages"],
            )

        cli_dependencies = all(
            name in self.installed_packages for name in ("pyyaml", "typer", "rich")
        )
        if repository_config["run_config_audit"] and cli_dependencies:
            config_result = self.command(
                [sys.executable, "-m", "catena.cli", "config-audit"],
                timeout=120,
                env=command_environment,
            )
            self.write_raw("config_audit.json", config_result)
            self.add_check(
                "repository_config_audit",
                "repository",
                PASS if config_result.passed else FAIL,
                (
                    "repository config audit passed"
                    if config_result.passed
                    else "repository config audit failed"
                ),
                observed=asdict(config_result),
            )
        elif repository_config["run_config_audit"]:
            self.add_check(
                "repository_config_audit",
                "repository",
                BLOCKED,
                "config audit blocked because PyYAML/Typer/Rich are unavailable",
                blocked_by=["python_packages"],
            )

        if repository_config["run_mock_smoke"] and cli_dependencies:
            smoke_result = self.command(
                [sys.executable, "-m", "catena.cli", "smoke"],
                timeout=120,
                env=command_environment,
            )
            self.write_raw("mock_smoke.json", smoke_result)
            self.add_check(
                "repository_mock_smoke",
                "repository",
                PASS if smoke_result.passed else FAIL,
                (
                    "repository mock smoke passed"
                    if smoke_result.passed
                    else "repository mock smoke failed"
                ),
                observed=asdict(smoke_result),
            )
        elif repository_config["run_mock_smoke"]:
            self.add_check(
                "repository_mock_smoke",
                "repository",
                BLOCKED,
                "mock smoke blocked because PyYAML/Typer/Rich are unavailable",
                blocked_by=["python_packages"],
            )

        repository_ids = {
            "git_snapshot",
            "python_compileall",
            "shell_syntax",
            "repository_pytest",
            "repository_config_audit",
            "repository_mock_smoke",
        }
        repository_checks = [
            check for check in self.checks if check.check_id in repository_ids
        ]
        repository_failures = [
            check
            for check in repository_checks
            if check.required and check.status in HARD_FAILURE_STATUSES
        ]
        self.add_check(
            "repository_validation",
            "repository",
            PASS if not repository_failures else BLOCKED,
            (
                "all repository validation gates passed"
                if not repository_failures
                else "repository validation is incomplete or failed"
            ),
            observed={
                "subchecks": [
                    {"check_id": check.check_id, "status": check.status}
                    for check in repository_checks
                ]
            },
            blocked_by=[check.check_id for check in repository_failures],
        )

    def _public_gpu_rows(self) -> list[dict[str, Any]]:
        cuda_by_index = {
            item["physical_index"]: item for item in self.cuda_lane_results
        }
        torch_by_index = {
            item["physical_index"]: item for item in self.torch_lane_results
        }
        rows: list[dict[str, Any]] = []
        for device in self.selected_gpus:
            index = device["physical_index"]
            rows.append(
                {
                    "physical_index": index,
                    "name": device["name"],
                    "memory_mib": device["memory_mib"],
                    "compute_capability": device["compute_capability"],
                    "mig_mode": device["mig_mode"],
                    "cuda_bf16": cuda_by_index.get(index, {}).get("passed"),
                    "pytorch_bf16": torch_by_index.get(index, {}).get("passed"),
                }
            )
        return rows

    def _ensure_canonical_checks(self) -> None:
        configured = list(self.config["checks"])
        present = {check.check_id for check in self.checks}
        for check_id in configured:
            if check_id == "reproducibility_manifest" or check_id in present:
                continue
            self.add_check(
                check_id,
                "audit_integrity",
                ERROR,
                f"canonical gate {check_id} was not produced",
                blocked_by=["audit_phase_exception"],
            )

    def _prepare_reproducibility_gate(self) -> None:
        if self.check("reproducibility_manifest") is not None:
            return
        self._ensure_canonical_checks()
        try:
            current_config_bytes = self.config_path.read_bytes()
            current_config_sha256 = hashlib.sha256(current_config_bytes).hexdigest()
            config_unchanged = current_config_bytes == self.config_source_bytes
            config_error = None
        except OSError as exc:
            current_config_sha256 = None
            config_unchanged = False
            config_error = repr(exc)
        resolved_path = self.run_dir / "resolved_config.yaml"
        resolved_sha256 = (
            _sha256_file(resolved_path) if resolved_path.is_file() else None
        )
        expected_resolved_sha256 = hashlib.sha256(
            self.resolved_config_text.encode("utf-8")
        ).hexdigest()
        resolved_matches = resolved_sha256 == expected_resolved_sha256
        canonical_before_manifest = set(self.config["checks"]) - {
            "reproducibility_manifest"
        }
        present = {check.check_id for check in self.checks}
        canonical_complete = canonical_before_manifest.issubset(present)
        source_snapshot_ready = bool(self.git.get("source_tree_sha256"))
        reproducibility_ready = (
            config_unchanged
            and resolved_matches
            and canonical_complete
            and source_snapshot_ready
        )
        self.add_check(
            "config_snapshot_immutable",
            "reproducibility",
            PASS if config_unchanged and resolved_matches else FAIL,
            (
                "source and resolved config snapshots remained immutable"
                if config_unchanged and resolved_matches
                else "config source changed or resolved snapshot is inconsistent"
            ),
            expected={
                "source_sha256": self.config_source_sha256,
                "resolved_sha256": expected_resolved_sha256,
            },
            observed={
                "current_source_sha256": current_config_sha256,
                "resolved_sha256": resolved_sha256,
                "read_error": config_error,
            },
        )
        self.add_check(
            "reproducibility_manifest",
            "reproducibility",
            PASS if reproducibility_ready else FAIL,
            (
                "immutable config, canonical gates, and source snapshot are ready "
                "for final artifact hashing"
                if reproducibility_ready
                else "reproducibility inputs are incomplete or inconsistent"
            ),
            expected={
                "canonical_check_ids": sorted(canonical_before_manifest),
                "config_unchanged": True,
                "source_tree_sha256": True,
            },
            observed={
                "canonical_complete": canonical_complete,
                "config_unchanged": config_unchanged,
                "resolved_matches": resolved_matches,
                "source_snapshot_ready": source_snapshot_ready,
            },
        )

    def finalize(self) -> dict[str, Any]:
        self._prepare_reproducibility_gate()
        failed = [
            check
            for check in self.checks
            if check.required and check.status in HARD_FAILURE_STATUSES
        ]
        warnings = [check for check in self.checks if check.status == WARN]
        passed = not failed
        host_count = len(self.host_gpus)
        selected_indices = list(self.config["hardware"]["selected_physical_gpus"])
        plan_changes: list[str] = []
        selected_inventory_complete = len(self.selected_gpus) == len(selected_indices)
        if (
            selected_inventory_complete
            and host_count
            and host_count != len(selected_indices)
        ):
            plan_changes.append(
                f"Pin E01-E12 to physical GPUs {selected_indices}; the host exposes "
                f"{host_count} GPUs rather than the four assumed by the plan."
            )
        storage_warning = next(
            (check for check in warnings if check.check_id == "storage_capacity"), None
        )
        if storage_warning is not None:
            plan_changes.append(
                "Resolve repository-local storage capacity before model downloads or "
                "teacher/checkpoint generation."
            )
        if failed:
            interpretation = (
                "E00 did not pass. E01 and all scientific experiments remain blocked until "
                "every hard failure is resolved and E00 is rerun. This infrastructure "
                "failure is not evidence for or against H1-H4."
            )
            plan_changes.insert(
                0,
                "Do not start E01; resolve all E00 hard failures and obtain a PASS rerun.",
            )
        else:
            interpretation = (
                "E00 passed. The environment is suitable for E01 runtime-adapter gates; "
                "this is an infrastructure result, not evidence for H1-H4."
            )
        if not plan_changes and passed:
            plan_changes = ["No experiment-plan change is required; proceed to E01."]

        report: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "e00_audit",
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "passed": passed,
            "status": PASS if passed else FAIL,
            "repository_root": ".",
            "artifact_dir": self.run_dir.relative_to(self.root).as_posix(),
            "config_path": self.config_path.relative_to(self.root).as_posix(),
            "config_sha256": self.config_source_sha256,
            "resolved_config_sha256": hashlib.sha256(
                self.resolved_config_text.encode("utf-8")
            ).hexdigest(),
            "checks": [asdict(check) for check in self.checks],
            "failed_check_ids": [check.check_id for check in failed],
            "warning_check_ids": [check.check_id for check in warnings],
            "environment": {
                "conda_name": os.getenv("CONDA_DEFAULT_ENV"),
                "conda_prefix_name": (
                    Path(os.environ["CONDA_PREFIX"]).name
                    if os.getenv("CONDA_PREFIX")
                    else None
                ),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "libc": platform.libc_ver(),
            },
            "phase_errors": self.phase_errors,
            "git": self.git,
            "host_gpu_count": host_count,
            "selected_physical_gpus": selected_indices,
            "selected_gpus": self.selected_gpus,
            "public_gpu_rows": self._public_gpu_rows(),
            "cuda_lane_results": self.cuda_lane_results,
            "torch_lane_results": self.torch_lane_results,
            "storage": self.storage_results,
            "interpretation": interpretation,
            "scientific_plan_change": "none",
            "plan_changes": plan_changes,
        }
        report_path = self.run_dir / "report.json"
        markdown_path = self.run_dir / "report.md"
        _atomic_write_json(report_path, report)
        _atomic_write_text(markdown_path, render_e00_markdown(report))

        initial_hashes: dict[str, str] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name not in {"SHA256SUMS", "manifest.json"}:
                initial_hashes[path.relative_to(self.run_dir).as_posix()] = _sha256_file(path)
        manifest = {
            "run_id": self.run_id,
            "report": "report.json",
            "report_sha256": _sha256_file(report_path),
            "config_sha256": report["config_sha256"],
            "resolved_config_sha256": report["resolved_config_sha256"],
            "source_tree_sha256": self.git.get("source_tree_sha256"),
            "files": initial_hashes,
        }
        _atomic_write_json(self.run_dir / "manifest.json", manifest)

        final_hashes: dict[str, str] = {}
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                final_hashes[path.relative_to(self.run_dir).as_posix()] = _sha256_file(path)
        checksum_text = "".join(
            f"{digest}  {relative}\n" for relative, digest in sorted(final_hashes.items())
        )
        _atomic_write_text(self.run_dir / "SHA256SUMS", checksum_text)

        latest = {
            "run_id": self.run_id,
            "passed": passed,
            "artifact_dir": report["artifact_dir"],
            "report_sha256": _sha256_file(report_path),
            "manifest_sha256": _sha256_file(self.run_dir / "manifest.json"),
        }
        _atomic_write_json(self.output_base / "latest.json", latest)
        if passed:
            _atomic_write_json(self.output_base / "latest_passed.json", latest)
        return report

    def _run_phase(
        self,
        phase_name: str,
        phase: Any,
        canonical_check_ids: Sequence[str],
    ) -> None:
        try:
            phase()
        except Exception as exc:  # noqa: BLE001 - phase failures must become artifacts.
            diagnostic = {
                "phase": phase_name,
                "exception_type": type(exc).__name__,
                "error": str(exc),
            }
            self.phase_errors.append(diagnostic)
            self.write_raw(
                f"phase_error_{phase_name}.json",
                json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n",
            )
            present = {check.check_id for check in self.checks}
            for check_id in canonical_check_ids:
                if check_id not in present:
                    self.add_check(
                        check_id,
                        phase_name,
                        ERROR,
                        f"{phase_name} raised {type(exc).__name__}: {exc}",
                        blocked_by=[f"{phase_name}_exception"],
                    )

    def run(self) -> dict[str, Any]:
        phases = [
            (
                "environment",
                self.audit_identity_and_packages,
                ("conda_environment", "python_runtime", "python_packages"),
            ),
            (
                "host_gpu",
                self.audit_host_gpu_inventory,
                ("host_gpu_inventory",),
            ),
            (
                "cuda_toolchain",
                self.audit_cuda_toolchain_and_lanes,
                ("cuda_toolchain", "cuda_bf16_lanes"),
            ),
            (
                "pytorch",
                self.audit_pytorch,
                ("pytorch_cuda", "pytorch_bf16_lanes"),
            ),
            ("storage", self.audit_storage, ("state_cache_storage",)),
            (
                "repository",
                self.audit_repository,
                ("repository_validation",),
            ),
        ]
        for phase_name, phase, check_ids in phases:
            self._run_phase(phase_name, phase, check_ids)
        return self.finalize()


def render_e00_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# E00 environment audit",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Status: **{report['status']}**",
        f"- Git: `{report.get('git', {}).get('commit') or 'unavailable'}`",
        f"- Artifact: `{report['artifact_dir']}`",
        "",
        "## Gate summary",
        "",
        "| Gate | Status | Result |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        if check["status"] == WARN or check["required"]:
            summary = str(check["summary"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{check['check_id']}` | {check['status']} | {summary} |")

    rows = report.get("public_gpu_rows") or []
    if rows:
        lines.extend(
            [
                "",
                "## Selected GPU lanes",
                "",
                "| GPU | Model | VRAM MiB | CC | MIG | CUDA BF16 | PyTorch BF16 |",
                "|---:|---|---:|---:|---|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['physical_index']} | {row['name']} | {row['memory_mib']} | "
                f"{row['compute_capability']} | {row['mig_mode']} | "
                f"{row['cuda_bf16']} | {row['pytorch_bf16']} |"
            )

    storage = report.get("storage") or {}
    if storage:
        lines.extend(
            [
                "",
                "## Repository-local storage",
                "",
                f"- Free before probe: {storage.get('free_gib_before', 0):.2f} GiB",
                f"- Median write: {storage.get('median_write_mib_s') or 0:.1f} MiB/s",
                f"- Median read: {storage.get('median_read_mib_s') or 0:.1f} MiB/s",
                "- Read figures may include page-cache effects; integrity is the hard gate.",
            ]
        )

    lines.extend(["", "## Interpretation", "", str(report["interpretation"])])
    lines.extend(["", "## Experiment-plan impact", ""])
    for change in report["plan_changes"]:
        lines.append(f"- {change}")
    lines.append("")
    return "\n".join(lines)


def run_e00_audit(
    config_path: str | Path = "configs/experiments/e00_audit.yaml",
    root: str | Path = ".",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    canonical_root = Path(__file__).resolve().parents[3]
    if root_path != canonical_root:
        raise E00ConfigError(
            f"E00 may only operate on the CATENA repository root: {canonical_root}"
        )
    config_file = _resolve_repo_path(root_path, str(config_path), "config path")
    try:
        config_source_bytes = config_file.read_bytes()
        config_text = config_source_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise E00ConfigError(f"could not read E00 config: {exc}") from exc
    config = _parse_e00_config_text(config_text, root_path)
    return E00Auditor(
        root_path,
        config_file,
        config,
        config_source_bytes=config_source_bytes,
    ).run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CATENA E00 hard-gate audit")
    parser.add_argument(
        "--config", default="configs/experiments/e00_audit.yaml", help="E00 config path"
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="write a failed report but return zero (diagnostic automation only)",
    )
    arguments = parser.parse_args(argv)
    try:
        report = run_e00_audit(arguments.config, arguments.root)
    except E00ConfigError as exc:
        print(json.dumps({"status": ERROR, "error": str(exc)}, ensure_ascii=False))
        return 2
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable CLI failure.
        print(
            json.dumps(
                {
                    "status": ERROR,
                    "error": f"unhandled E00 finalization error: {type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "status": report["status"],
                "artifact_dir": report["artifact_dir"],
                "failed_check_ids": report["failed_check_ids"],
                "warning_check_ids": report["warning_check_ids"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] or arguments.allow_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
