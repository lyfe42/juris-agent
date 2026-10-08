from __future__ import annotations

import pytest


class FakeClock:
    """可推进的假时钟。测试不真的 sleep：整套执行层测试跑在毫秒级。"""

    def __init__(self, start: float = 1_000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class RecordingSleeper:
    """记录 sleep 时长但不真的等待，用于断言退避序列。"""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeper() -> RecordingSleeper:
    return RecordingSleeper()
