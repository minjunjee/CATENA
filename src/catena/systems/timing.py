from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class TimedResult:
    value: object
    seconds: float


def timed_call(function: Callable[[], T]) -> TimedResult:
    start = time.perf_counter()
    value = function()
    return TimedResult(value=value, seconds=time.perf_counter() - start)
