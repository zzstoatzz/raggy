"""Lightweight performance instrumentation helpers for the Prefect docs MCP."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")

    if len(data) == 1:
        return float(data[0])

    ordered = sorted(data)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class _StageRecorder:
    """Collects numeric samples for a single metric stage."""

    def __init__(self, maxlen: int) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            self._samples.append(float(value))

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def snapshot(self) -> Dict[str, float | int]:
        with self._lock:
            data = list(self._samples)

        if not data:
            return {"count": 0}

        ordered = sorted(data)
        return {
            "count": len(ordered),
            "min_ms": ordered[0] * 1000,
            "max_ms": ordered[-1] * 1000,
            "mean_ms": statistics.fmean(ordered) * 1000,
            "median_ms": statistics.median(ordered) * 1000,
            "p95_ms": _percentile(ordered, 0.95) * 1000,
        }


class PerformanceMetrics:
    """Thread-safe collector for recent performance measurements."""

    def __init__(self, max_samples: int = 200) -> None:
        self._max_samples = max_samples
        self._stages: dict[str, _StageRecorder] = {}
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def _get_stage(self, stage: str) -> _StageRecorder:
        with self._lock:
            recorder = self._stages.get(stage)
            if recorder is None:
                recorder = _StageRecorder(self._max_samples)
                self._stages[stage] = recorder
        return recorder

    def record(self, stage: str, value: float) -> None:
        self._get_stage(stage).add(value)

    def increment(self, counter: str) -> None:
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + 1

    def reset(self) -> None:
        with self._lock:
            for recorder in self._stages.values():
                recorder.reset()
            self._counters.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            stages = dict(self._stages)

        return {
            "stages": {name: recorder.snapshot() for name, recorder in stages.items()},
            "counters": counters,
        }

    @contextmanager
    def timer(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - start)


# Single module-level instance used by the server and diagnostics scripts.
metrics = PerformanceMetrics()


__all__ = ["PerformanceMetrics", "metrics"]

