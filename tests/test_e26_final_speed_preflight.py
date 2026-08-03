from __future__ import annotations

from pathlib import Path
from types import ModuleType

import torch

from tools.run_e26_final_speed_preflight import (
    ChunkDispatchCounter,
    NvidiaSmiSampler,
    _worker_environment,
)


def test_chunk_counter_observes_exact_symbol_and_restores() -> None:
    module = ModuleType("lit_gpt.gdn2")

    def chunk(value: torch.Tensor) -> torch.Tensor:
        return value.square()

    module.__dict__["chunk_gdn2"] = chunk
    value = torch.tensor([2.0])
    with ChunkDispatchCounter(module) as counter:
        assert torch.equal(module.chunk_gdn2(value), torch.tensor([4.0]))
        assert (counter.calls, counter.completed) == (1, 1)
    assert module.chunk_gdn2 is chunk


def test_nvidia_smi_row_parser_is_strict() -> None:
    assert NvidiaSmiSampler.parse_row("75, 310.5, 2048\n") == {
        "utilization_percent": 75.0,
        "power_watts": 310.5,
        "memory_used_mib": 2048.0,
    }


def test_worker_environment_locks_imports_and_disables_bytecode(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    fla = tmp_path / "fla"
    for path in (repository, runtime, fla):
        path.mkdir()
    environment = _worker_environment(repository, runtime, fla, 3)
    roots = environment["PYTHONPATH"].split(":")
    assert roots[:4] == [
        str(repository / "src"),
        str(repository),
        str(runtime),
        str(fla),
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
