from __future__ import annotations

import importlib
import math
import re
import subprocess
from pathlib import Path
from typing import Any

_FULL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_UNRESOLVED_ENV_PATTERN = re.compile(r"^\$\{[^}]+\}$")


class OfficialBackendNotConfigured(RuntimeError):
    """Raised when an official backend dependency has not been provided."""


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _require_configured_text(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not text or _UNRESOLVED_ENV_PATTERN.fullmatch(text):
        raise OfficialBackendNotConfigured(f"{label} is not configured")
    return text


def _compare_metric(value: object, specification: dict[str, Any]) -> bool:
    comparison = str(specification["comparison"])
    if comparison == "eq":
        return value == specification["expected"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"numeric check received a non-numeric metric: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"official metric is non-finite: {numeric!r}")
    threshold = float(specification["threshold"])
    if comparison == "le":
        return numeric <= threshold
    if comparison == "ge":
        return numeric >= threshold
    raise ValueError(f"unsupported official metric comparison: {comparison!r}")


def evaluate_metric_checks(
    *,
    metrics: dict[str, Any],
    check_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    if not check_contract:
        raise ValueError("official backend check contract cannot be empty")
    checks: dict[str, dict[str, Any]] = {}
    for check_name, raw_specification in check_contract.items():
        if not isinstance(raw_specification, dict):
            raise TypeError(f"check {check_name!r} must be a mapping")
        specification = dict(raw_specification)
        metric_key = str(specification["metric_key"])
        if metric_key not in metrics:
            raise KeyError(
                f"official plugin did not return required metric {metric_key!r}"
            )
        value = metrics[metric_key]
        passed = _compare_metric(value, specification)
        checks[str(check_name)] = {
            "metric_key": metric_key,
            "value": value,
            "comparison": str(specification["comparison"]),
            "threshold": specification.get("threshold"),
            "expected": specification.get("expected"),
            "passed": passed,
        }
    return checks, all(item["passed"] for item in checks.values())


def run_official_operator_gate(
    *,
    backend: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    name = str(backend["name"])
    base = {
        "name": name,
        "repo_path": str(backend["repo_path"]),
        "plugin_module": str(backend["plugin_module"]),
    }
    if dry_run:
        return {
            **base,
            "status": "DRY_RUN",
            "configured": False,
            "scientific_evidence": False,
            "checks": {},
        }

    try:
        repo_text = _require_configured_text(
            backend["repo_path"],
            label=f"{name}.repo_path",
        )
        commit = _require_configured_text(
            backend["expected_commit"],
            label=f"{name}.expected_commit",
        )
        plugin_module = _require_configured_text(
            backend["plugin_module"],
            label=f"{name}.plugin_module",
        )
        if not _FULL_COMMIT_PATTERN.fullmatch(commit):
            raise ValueError(
                f"{name}.expected_commit must be a full lowercase commit SHA"
            )
        repo_path = Path(repo_text)
        if not repo_path.is_dir():
            raise OfficialBackendNotConfigured(
                f"official backend repository does not exist: {repo_path}"
            )
        actual_commit = _git_head(repo_path)
        if actual_commit != commit:
            raise ValueError(
                f"commit mismatch for {name}: "
                f"expected={commit!r}, actual={actual_commit!r}"
            )
        try:
            module = importlib.import_module(plugin_module)
        except ModuleNotFoundError as error:
            raise OfficialBackendNotConfigured(
                f"official plugin is unavailable: {plugin_module}"
            ) from error
        if not hasattr(module, "run_backend_gate"):
            raise AttributeError(
                f"{plugin_module} must expose run_backend_gate(config: dict)"
            )
        result = module.run_backend_gate(dict(backend))
        if not isinstance(result, dict):
            raise TypeError("official backend plugin must return a dictionary")
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("official backend plugin result.metrics must be a mapping")
        declared_scientific = result.get("scientific_evidence")
        if not isinstance(declared_scientific, bool):
            raise TypeError(
                "official backend plugin must explicitly return "
                "scientific_evidence as a boolean"
            )
        checks, checks_passed = evaluate_metric_checks(
            metrics=dict(metrics),
            check_contract=dict(backend["checks"]),
        )
        declared_passed = result.get("passed")
        if declared_passed is not None and declared_passed is not checks_passed:
            raise ValueError(
                "plugin result.passed disagrees with authoritative metric checks"
            )
        scientific_evidence = bool(checks_passed and declared_scientific)
        return {
            **base,
            "status": "PASS" if scientific_evidence else "FAIL",
            "configured": True,
            "commit": actual_commit,
            "metrics": metrics,
            "checks": checks,
            "scientific_evidence": scientific_evidence,
        }
    except OfficialBackendNotConfigured as error:
        return {
            **base,
            "status": "NOT_CONFIGURED",
            "configured": False,
            "error": f"{type(error).__name__}: {error}",
            "scientific_evidence": False,
            "checks": {},
        }
    except Exception as error:  # strict gate: configured failures never fall back
        return {
            **base,
            "status": "FAIL",
            "configured": True,
            "error": f"{type(error).__name__}: {error}",
            "scientific_evidence": False,
            "checks": {},
        }

