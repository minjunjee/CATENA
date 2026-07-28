"""Strict source/environment boundary for the E25a official gate."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catena.core.io import file_sha256

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_UNRESOLVED = re.compile(r"^\$\{[^}]+\}$")
_REPO_ROOT = Path(__file__).resolve().parents[3]


class E25aNotConfigured(RuntimeError):
    """The pinned official-only runtime is incomplete."""


def _text(value: object, label: str) -> str:
    result = str(value).strip()
    if not result or _UNRESOLVED.fullmatch(result):
        raise E25aNotConfigured(f"{label} is not configured")
    return result


def _git(path: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise E25aNotConfigured(f"cannot inspect official source {path}: {error}") from error


def _repository_manifest(
    *,
    path: object,
    expected_commit: object,
    expected_remote: object | None,
    license_relative: object,
    license_sha256: object,
    label: str,
) -> dict[str, Any]:
    repository = Path(_text(path, f"{label}.path")).resolve()
    if not repository.is_dir():
        raise E25aNotConfigured(f"{label} repository is missing: {repository}")
    commit = _text(expected_commit, f"{label}.expected_commit")
    if not _COMMIT.fullmatch(commit):
        raise ValueError(f"{label} requires a full lowercase Git commit")
    actual = _git(repository, "rev-parse", "HEAD")
    if actual != commit:
        raise ValueError(f"{label} commit mismatch: expected={commit}, actual={actual}")
    tracked_dirty = _git(repository, "status", "--porcelain", "--untracked-files=no")
    if tracked_dirty:
        raise ValueError(f"{label} tracked source is dirty")
    remote = _git(repository, "remote", "get-url", "origin")
    if expected_remote is not None and remote != str(expected_remote):
        raise ValueError(f"{label} remote mismatch: {remote}")
    license_path = (repository / _text(license_relative, f"{label}.license_path")).resolve()
    try:
        license_path.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{label} license path escapes repository") from error
    expected_license = _text(license_sha256, f"{label}.license_sha256")
    actual_license = file_sha256(license_path)
    if actual_license != expected_license:
        raise ValueError(f"{label} license hash mismatch")
    return {
        "path": str(repository),
        "commit": actual,
        "remote": remote,
        "tracked_source_clean": True,
        "license": {
            "path": str(license_path),
            "sha256": actual_license,
        },
    }


def _plugin_manifest(
    *,
    module_name: object,
    source_path: object,
    source_sha256: object,
    label: str,
    relative_to_repo: bool = False,
) -> dict[str, str]:
    raw_path = Path(_text(source_path, f"{label}.source_path"))
    plugin_path = (
        (_REPO_ROOT / raw_path).resolve()
        if relative_to_repo and not raw_path.is_absolute()
        else raw_path.resolve()
    )
    if not plugin_path.is_file() or plugin_path.is_symlink():
        raise E25aNotConfigured(f"{label} plugin is missing or unsafe: {plugin_path}")
    if relative_to_repo:
        try:
            plugin_path.relative_to(_REPO_ROOT)
        except ValueError as error:
            raise ValueError(f"{label} plugin path escapes the repository") from error
    plugin_hash = file_sha256(plugin_path)
    expected_plugin_hash = _text(source_sha256, f"{label}.source_sha256")
    if plugin_hash != expected_plugin_hash:
        raise ValueError(f"{label} plugin hash mismatch")
    resolved_module = _text(module_name, f"{label}.module")
    specification = importlib.util.find_spec(resolved_module)
    if specification is None or specification.origin is None:
        raise E25aNotConfigured(f"{label} plugin cannot be imported: {resolved_module}")
    origin = Path(specification.origin).resolve()
    if origin != plugin_path:
        raise ValueError(f"{label} plugin origin mismatch: expected={plugin_path}, actual={origin}")
    return {
        "module": resolved_module,
        "path": str(plugin_path),
        "sha256": plugin_hash,
    }


def official_source_manifest(
    config: Mapping[str, Any],
    *,
    dry_run: bool,
    stage: str = "gate",
) -> dict[str, Any]:
    """Validate environment, source, license and plugin provenance."""

    if stage not in {"gate", "replication"}:
        raise ValueError(f"unsupported E25a stage: {stage}")
    if dry_run:
        return {
            "status": "DRY_RUN",
            "configured": False,
            "reference_fallback": False,
            "current_python_prefix": str(Path(sys.prefix).resolve()),
        }
    environment = config["environment"]
    required_prefix = Path(
        _text(environment["required_prefix"], "environment.required_prefix")
    ).resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix != required_prefix:
        raise E25aNotConfigured(
            f"E25a must run in {required_prefix}, current prefix is {actual_prefix}"
        )
    backend = config["backend"]
    gdn2 = _repository_manifest(
        path=backend["repo_path"],
        expected_commit=backend["expected_commit"],
        expected_remote=backend["remote_url"],
        license_relative=backend["license_path"],
        license_sha256=backend["license_sha256"],
        label="GDN2",
    )
    fla = _repository_manifest(
        path=backend["fla_repo_path"],
        expected_commit=backend["fla_expected_commit"],
        expected_remote=None,
        license_relative=backend["fla_license_path"],
        license_sha256=backend["fla_license_sha256"],
        label="FLA",
    )
    gate_plugin = _plugin_manifest(
        module_name=backend["plugin_module"],
        source_path=backend["plugin_source_path"],
        source_sha256=backend["plugin_source_sha256"],
        label="official adapter",
    )
    manifest: dict[str, Any] = {
        "status": "CONFIGURED",
        "configured": True,
        "reference_fallback": False,
        "python_prefix": str(actual_prefix),
        "gdn2": gdn2,
        "fla": fla,
        "plugin": gate_plugin,
    }
    if stage == "replication":
        replication = config["replication"]
        manifest["replication_plugin"] = _plugin_manifest(
            module_name=replication["plugin_module"],
            source_path=replication["plugin_source_path"],
            source_sha256=replication["plugin_source_sha256"],
            label="official replication",
            relative_to_repo=True,
        )
    return manifest


def validate_gate_dependency(
    path: Path,
    *,
    expected_experiment_id: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise E25aNotConfigured(f"explicit gate report is missing or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gate report must be a JSON object")
    if (
        payload.get("experiment_id") != expected_experiment_id
        or payload.get("execution_status") != "PASS"
        or payload.get("stage") != "GATE"
        or payload.get("official_operator_gate_passed") is not True
        or payload.get("scientific_evidence") is not True
    ):
        raise ValueError("explicit E25a gate report did not pass the official-only gate")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "report": payload,
    }


def environment_defaults() -> dict[str, str]:
    """Document the existing local separate environment without mutating it."""

    return {
        "CATENA_E25A_ENV_PREFIX": os.environ.get("CATENA_E25A_ENV_PREFIX", ""),
        "CATENA_GDN2_REPO": os.environ.get("CATENA_GDN2_REPO", ""),
        "CATENA_FLA_REPO": os.environ.get("CATENA_FLA_REPO", ""),
        "CATENA_E25A_PLUGIN_SOURCE": os.environ.get("CATENA_E25A_PLUGIN_SOURCE", ""),
    }
