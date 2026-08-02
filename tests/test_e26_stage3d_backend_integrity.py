from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location(
    "catena_e26_stage3d_backend_tool",
    Path(__file__).resolve().parents[1] / "tools" / "run_e26_stage3d_preflight.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


def _optimized() -> dict[str, Any]:
    return {
        "backend_id": "compiled_scan",
        "autocast_dtype": "torch.bfloat16",
        "backend_metadata": {
            "python_token_loop_at_runtime": False,
            "accumulation_policy": "locked-identical-policy",
        },
        "backend_diagnostics": {
            "graph_invocations": 2,
            "optimized_calls": 1,
            "chunks_executed": 128,
            "padded_tokens": 0,
            "fallback_count": 0,
            "graph_break_count": 0,
        },
    }


def test_backend_integrity_requires_strict_reference_and_compiled_execution() -> None:
    result = _TOOL._backend_integrity_from_diagnostics(
        optimized=_optimized(),
        reference_backend_id="reference_python",
    )
    assert result["passed"] is True
    assert result["strict_reference_python"] is True
    assert result["registered_backend_id"] == "torch_compile_fixed_chunk_scan_v1"
    assert result["runtime_backend_alias"] == "compiled_scan"
    assert result["backend_alias_matches_registration"] is True
    assert result["python_token_loop_at_scientific_runtime"] is False
    assert result["fallback_count"] == 0
    assert result["graph_break_count"] == 0


def test_backend_integrity_fails_on_fallback_graphbreak_or_python_path() -> None:
    for mutation in (
        {"backend_diagnostics": {**_optimized()["backend_diagnostics"], "fallback_count": 1}},
        {"backend_diagnostics": {**_optimized()["backend_diagnostics"], "graph_break_count": 1}},
        {
            "backend_metadata": {
                **_optimized()["backend_metadata"],
                "python_token_loop_at_runtime": True,
            }
        },
    ):
        optimized = {**_optimized(), **mutation}
        assert (
            _TOOL._backend_integrity_from_diagnostics(
                optimized=optimized,
                reference_backend_id="reference_python",
            )["passed"]
            is False
        )
    assert (
        _TOOL._backend_integrity_from_diagnostics(
            optimized=_optimized(),
            reference_backend_id="compiled_scan",
        )["passed"]
        is False
    )
    assert (
        _TOOL._backend_integrity_from_diagnostics(
            optimized={**_optimized(), "backend_id": "unregistered_alias"},
            reference_backend_id="reference_python",
        )["passed"]
        is False
    )


def test_backend_integrity_rejects_variant_specific_precision_or_padding() -> None:
    assert (
        _TOOL._backend_integrity_from_diagnostics(
            optimized=_optimized(),
            reference_backend_id="reference_python",
            variant_specific_fp32_path_count=1,
        )["passed"]
        is False
    )
    assert (
        _TOOL._backend_integrity_from_diagnostics(
            optimized=_optimized(),
            reference_backend_id="reference_python",
            variant_specific_padding_count=1,
        )["passed"]
        is False
    )


def _g3_row(variant: str, *, token_sha: str = "1" * 64) -> dict[str, Any]:
    backend = _TOOL._backend_integrity_from_diagnostics(
        optimized=_optimized(),
        reference_backend_id="reference_python",
    )
    return {
        "candidate_id": "d512_ctx4096",
        "variant": variant,
        "state_context": "zero_state",
        "fixed_layout": {"candidate_id": "d512_ctx4096"},
        "initialization_digest": "2" * 64,
        "parameter_signature_sha256": "3" * 64,
        "optimizer_state_signature_sha256": "4" * 64,
        "token_ids_sha256": token_sha,
        "data_cursor_sha256": "5" * 64,
        "layout_identity_passed": True,
        "backend_integrity": backend,
        "variant_specific_padding_count": 0,
        "variant_specific_fp32_path_count": 0,
        "passed": True,
    }


def test_cross_variant_pair_enforces_data_identity_and_observed_policy() -> None:
    tied = _g3_row("projected_tied_delta_lm")
    dual = _g3_row("dual_delta_lm")
    rows = _TOOL._apply_cross_variant_g1_and_g6([tied, dual])
    assert all(row["layout_identity_passed"] is True for row in rows)
    assert all(row["passed"] is True for row in rows)

    drifted = copy.deepcopy(dual)
    drifted["token_ids_sha256"] = "9" * 64
    rows = _TOOL._apply_cross_variant_g1_and_g6([tied, drifted])
    assert all(row["layout_identity_passed"] is False for row in rows)
    assert all(row["passed"] is False for row in rows)
