import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional


@contextmanager
def stopwatch(sink: Optional[Callable[[str, float], None]] = None, label: str = "") -> Iterator[Callable[[], float]]:
    """Time a block, report elapsed seconds to sink on exit."""
    start = time.perf_counter()
    finished = []

    def elapsed() -> float:
        return finished[0] if finished else time.perf_counter() - start

    try:
        yield elapsed
    finally:
        finished.append(time.perf_counter() - start)
        if sink is not None:
            sink(label, finished[0])


class Deadline:
    """Monotonic wall clock budget shared by the guarded interpreter."""

    __slots__ = ("_end",)

    def __init__(self, seconds: float):
        self._end = time.perf_counter() + max(0.0, seconds)

    @property
    def expired(self) -> bool:
        return time.perf_counter() >= self._end

    @property
    def remaining(self) -> float:
        return max(0.0, self._end - time.perf_counter())
