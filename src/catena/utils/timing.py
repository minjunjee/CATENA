from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimingResult:
    milliseconds: float = 0.0


def synchronize_if_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


@contextmanager
def measured(result: TimingResult):
    synchronize_if_cuda()
    started = time.perf_counter()
    yield
    synchronize_if_cuda()
    result.milliseconds = (time.perf_counter() - started) * 1000.0
