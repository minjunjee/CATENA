from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from catena.config import ConfigError, RunPaths
from catena.experiments.e00_audit import (
    BLOCKED,
    ERROR,
    FAIL,
    PASS,
    E00Auditor,
    E00ConfigError,
    _parse_nvidia_inventory,
    _probe_identity_matches,
    _version_satisfies,
    load_e00_config,
    render_e00_markdown,
    run_command,
    run_e00_audit,
    validate_e00_config,
)
from catena.utils.manifest import write_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def valid_config() -> dict:
    return json.loads(
        (REPOSITORY_ROOT / "configs" / "experiments" / "e00_audit.yaml").read_text(
            encoding="utf-8"
        )
    )


class E00ConfigTests(unittest.TestCase):
    def test_repository_config_is_valid_and_json_compatible_yaml(self) -> None:
        config = load_e00_config(
            "configs/experiments/e00_audit.yaml", REPOSITORY_ROOT
        )
        self.assertEqual(config["experiment"], "e00_audit")
        self.assertEqual(validate_e00_config(config, REPOSITORY_ROOT), [])

    def test_unknown_key_is_rejected(self) -> None:
        config = valid_config()
        config["unexpected"] = True
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("unknown top-level keys" in error for error in errors))

    def test_mandatory_check_cannot_be_disabled(self) -> None:
        config = valid_config()
        config["checks"].remove("pytorch_bf16_lanes")
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("mandatory checks missing" in error for error in errors))

    def test_output_path_cannot_escape_repository(self) -> None:
        config = valid_config()
        config["output_dir"] = "../outside"
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("escapes repository root" in error for error in errors))

    def test_selected_gpu_count_must_match(self) -> None:
        config = valid_config()
        config["hardware"]["selected_physical_gpus"] = [0, 1, 2]
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("must equal selected_physical_gpus" in error for error in errors))

    def test_gpu_gate_is_exactly_four_and_rejects_boolean_indices(self) -> None:
        config = valid_config()
        config["hardware"]["selected_physical_gpus"] = [True, 1, 2, 3]
        config["hardware"]["expected_visible_gpu_count"] = True
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("list of indices" in error for error in errors))
        self.assertTrue(any("exactly 4" in error for error in errors))

    def test_repository_hard_gates_cannot_be_disabled(self) -> None:
        config = valid_config()
        config["repository"]["run_pytest"] = False
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("mandatory hard gate" in error for error in errors))

    def test_unknown_nested_key_is_rejected(self) -> None:
        config = valid_config()
        config["toolchain"]["surprise"] = 1
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("unknown toolchain keys" in error for error in errors))

    def test_all_declared_project_dependencies_are_audited(self) -> None:
        config = valid_config()
        del config["environment"]["required_packages"]["pydantic"]
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("omits pyproject dependencies" in error for error in errors))

    def test_declared_dependency_constraints_cannot_drift(self) -> None:
        config = valid_config()
        config["environment"]["required_packages"]["pydantic"] = ">=1"
        errors = validate_e00_config(config, REPOSITORY_ROOT)
        self.assertTrue(any("constraints differ" in error for error in errors))


class E00UtilityTests(unittest.TestCase):
    def test_version_constraints(self) -> None:
        self.assertTrue(_version_satisfies("3.11.15", ">=3.11,<3.12"))
        self.assertTrue(_version_satisfies("2.12.1+cu130", "==2.12.1+cu130"))
        self.assertFalse(_version_satisfies("2.12.0+cu130", "==2.12.1+cu130"))

    def test_nvidia_inventory_parser(self) -> None:
        row = (
            "0, NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-example, "
            "580.126.16, 97887, 00000000:03:00.0, 12.0, Disabled\n"
        )
        parsed = _parse_nvidia_inventory(row)
        self.assertEqual(parsed[0]["physical_index"], 0)
        self.assertEqual(parsed[0]["memory_mib"], 97887)
        self.assertEqual(parsed[0]["compute_capability"], "12.0")

    def test_command_timeout_is_reported(self) -> None:
        result = run_command(
            [
                sys.executable,
                "-c",
                "import time; print('started', flush=True); time.sleep(2)",
            ],
            cwd=REPOSITORY_ROOT,
            timeout=0.05,
        )
        self.assertTrue(result.timed_out)
        self.assertFalse(result.passed)
        self.assertIsInstance(result.stdout, str)
        self.assertIn("started", result.stdout)

    def test_markdown_contains_interpretation_and_plan_impact(self) -> None:
        markdown = render_e00_markdown(
            {
                "run_id": "test",
                "status": FAIL,
                "artifact_dir": "artifacts/test",
                "git": {"commit": "abc"},
                "checks": [
                    {
                        "check_id": "pytorch_cuda",
                        "status": BLOCKED,
                        "required": True,
                        "summary": "missing torch",
                    }
                ],
                "public_gpu_rows": [],
                "storage": {},
                "interpretation": "E01 is blocked.",
                "plan_changes": ["Install the approved runtime, then rerun E00."],
            }
        )
        self.assertIn("## Interpretation", markdown)
        self.assertIn("## Experiment-plan impact", markdown)
        self.assertIn("E01 is blocked", markdown)

    def test_run_paths_reject_output_escape(self) -> None:
        with self.assertRaises(ConfigError):
            RunPaths.from_config({"output_dir": "../outside"}, REPOSITORY_ROOT)

    def test_runtime_manifest_rejects_output_escape(self) -> None:
        with self.assertRaises(ValueError):
            write_manifest("../outside-catena-test")

    def test_device_identity_must_match_inventory(self) -> None:
        inventory = {
            "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "pci_bus_id": "00000000:03:00.0",
            "name": "GPU model",
            "compute_capability": "12.0",
        }
        probe = {
            "uuid": inventory["uuid"],
            "pci_bus_id": "0000:03:00.0",
            "name": inventory["name"],
            "compute_capability": "12.0",
            "visible_device_count": 1,
        }
        self.assertTrue(_probe_identity_matches(probe, inventory))
        probe["uuid"] = "GPU-wrong"
        self.assertFalse(_probe_identity_matches(probe, inventory))

    def test_arbitrary_repository_root_is_rejected(self) -> None:
        artifact_parent = REPOSITORY_ROOT / "artifacts" / "test_tmp"
        artifact_parent.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=artifact_parent) as temporary,
            self.assertRaises(E00ConfigError),
        ):
            run_e00_audit(root=temporary)


class E00ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact_parent = REPOSITORY_ROOT / "artifacts" / "test_tmp"
        artifact_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=artifact_parent)
        self.root = Path(self.temporary.name)
        config = valid_config()
        config["output_dir"] = "artifacts/e00"
        config["storage"] = {
            "path": "artifacts/storage_probe",
            "model_cache_path": ".scratch/model_cache_probe",
            "probe_size_mib": 1,
            "repeats": 1,
            "recommended_free_gib": 0,
        }
        self.config = config
        self.config_path = self.root / "e00.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_storage_probe_checks_hash_and_removes_probe_file(self) -> None:
        auditor = E00Auditor(self.root, self.config_path, self.config)
        auditor.audit_storage()
        check = next(item for item in auditor.checks if item.check_id == "state_cache_storage")
        self.assertEqual(check.status, PASS)
        probe_dir = self.root / "artifacts" / "storage_probe"
        self.assertEqual(list(probe_dir.glob("probe-*.bin")), [])

    def test_failed_gate_still_writes_report_and_does_not_write_latest_passed(self) -> None:
        auditor = E00Auditor(self.root, self.config_path, self.config)
        auditor.add_check("synthetic_failure", "test", FAIL, "expected test failure")
        report = auditor.finalize()
        self.assertFalse(report["passed"])
        self.assertTrue((auditor.run_dir / "report.json").is_file())
        self.assertTrue((auditor.run_dir / "SHA256SUMS").is_file())
        self.assertFalse((auditor.output_base / "latest_passed.json").exists())

    def test_finalize_inserts_missing_canonical_gates_as_errors(self) -> None:
        auditor = E00Auditor(self.root, self.config_path, self.config)
        report = auditor.finalize()
        report_checks = {item["check_id"]: item for item in report["checks"]}
        for check_id in self.config["checks"]:
            self.assertIn(check_id, report_checks)
        self.assertEqual(report_checks["conda_environment"]["status"], ERROR)
        self.assertFalse(report["passed"])

    def test_config_mutation_is_detected_and_initial_hash_is_preserved(self) -> None:
        auditor = E00Auditor(self.root, self.config_path, self.config)
        initial_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        changed = json.loads(self.config_path.read_text(encoding="utf-8"))
        changed["storage"]["recommended_free_gib"] = 1
        self.config_path.write_text(json.dumps(changed), encoding="utf-8")
        report = auditor.finalize()
        immutable = next(
            item
            for item in report["checks"]
            if item["check_id"] == "config_snapshot_immutable"
        )
        self.assertEqual(immutable["status"], FAIL)
        self.assertEqual(report["config_sha256"], initial_hash)

    def test_phase_exception_still_produces_failed_report(self) -> None:
        auditor = E00Auditor(self.root, self.config_path, self.config)

        def crash() -> None:
            raise RuntimeError("synthetic phase crash")

        auditor._run_phase("synthetic", crash, ("state_cache_storage",))
        report = auditor.finalize()
        self.assertFalse(report["passed"])
        self.assertEqual(report["phase_errors"][0]["phase"], "synthetic")
        self.assertTrue((auditor.run_dir / "report.json").is_file())
        self.assertTrue(
            (auditor.raw_dir / "phase_error_synthetic.json").is_file()
        )

    def test_all_canonical_gates_finalize_pass_and_checksums_verify(self) -> None:
        auditor = E00Auditor(self.root, self.config_path, self.config)
        for check_id in self.config["checks"]:
            if check_id != "reproducibility_manifest":
                auditor.add_check(check_id, "synthetic", PASS, "synthetic pass")
        auditor.git = {"source_tree_sha256": "abc123"}
        report = auditor.finalize()
        self.assertTrue(report["passed"])
        for line in (auditor.run_dir / "SHA256SUMS").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, relative = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((auditor.run_dir / relative).read_bytes()).hexdigest(),
                digest,
            )


if __name__ == "__main__":
    unittest.main()
