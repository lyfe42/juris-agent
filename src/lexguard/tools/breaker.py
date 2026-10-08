"""滑动窗口熔断器（每工具一个实例）。

三个状态与转移：

```
CLOSED    ──窗口内失败率 > 阈值且样本足够──> OPEN
OPEN      ──经过 open_seconds 冷却────────> HALF_OPEN
HALF_OPEN ──探测成功──> CLOSED（窗口清空）
HALF_OPEN ──探测失败──> OPEN（冷却重新计时）
```

三个容易做错的地方，测试各有一条：

1. **样本不足不熔断**。没有 `min_samples` 时，"第一个请求恰好失败"会让失败率
   立刻达到 100% 从而打开熔断——工具在冷启动阶段就被打死，且永远不会恢复。
2. **HALF_OPEN 只放行固定数量的探测**。若半开状态无限制放行，等于熔断没生效；
   若只放行一个，则一个瞬时抖动又会重新打开，造成长时间不可用。
3. **半开探测成功要清空窗口**。否则上一次故障的失败样本还留在滑窗里，
   下一个请求立刻又触发熔断，形成"打开—探测—打开"的抖动循环。

时钟是注入的（`clock` 参数），测试里用假时钟推进，不真的 sleep。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    window: int = 20
    min_samples: int = 5
    # 命名刻意带 _threshold：本类另有一个计算出来的 `failure_rate` 属性，
    # 两者同名会让 dataclass 的 `self.failure_ratio = ...` 撞上只读 property 而报错。
    failure_ratio_threshold: float = 0.5
    open_seconds: float = 30.0
    half_open_probes: int = 2
    clock: Callable[[], float] = time.monotonic

    state: CircuitState = CircuitState.CLOSED
    _outcomes: deque[bool] = field(default_factory=deque, repr=False, compare=False)
    _opened_at: float | None = field(default=None, repr=False, compare=False)
    _probes_left: int = field(default=0, repr=False, compare=False)
    # 累计计数器（不参与窗口、不随时间清零）。存在的理由是**让"熔断到底有没有生效"
    # 可被直接测量**：只看成功率时，一个永不触发的熔断器和没装熔断器完全无法区分。
    trips: int = 0
    rejections: int = 0

    def __post_init__(self) -> None:
        self._outcomes = deque(maxlen=self.window)

    # —— 只读观测 ——
    @property
    def samples(self) -> int:
        return len(self._outcomes)

    @property
    def failure_rate(self) -> float:
        """窗口内失败比例。空窗口记 0.0（不是 1.0——否则冷启动即熔断）。"""
        if not self._outcomes:
            return 0.0
        return 1.0 - (sum(self._outcomes) / len(self._outcomes))

    # —— 状态推进 ——
    def allow(self) -> tuple[bool, float]:
        """是否放行本次调用。返回 `(是否放行, 建议等待秒数)`。"""
        now = self.clock()

        if self.state is CircuitState.OPEN:
            assert self._opened_at is not None
            waited = now - self._opened_at
            if waited >= self.open_seconds:
                self.state = CircuitState.HALF_OPEN
                self._probes_left = self.half_open_probes
            else:
                self.rejections += 1
                return False, self.open_seconds - waited

        if self.state is CircuitState.HALF_OPEN:
            if self._probes_left <= 0:
                # 探测额度用尽仍未判定成功（并发场景），继续拒绝
                self.rejections += 1
                return False, self.open_seconds
            self._probes_left -= 1

        return True, 0.0

    def record(self, success: bool) -> None:
        """记录一次**真实发起了**的调用结果。被熔断拒绝的调用不能记进来。"""
        if self.state is CircuitState.HALF_OPEN:
            if success:
                self.state = CircuitState.CLOSED
                self._outcomes.clear()  # 关键：清掉历史失败样本
                self._opened_at = None
                self._probes_left = 0
            else:
                self.state = CircuitState.OPEN
                self._opened_at = self.clock()
                self._probes_left = 0
                self.trips += 1
            return

        self._outcomes.append(success)
        if (
            self.state is CircuitState.CLOSED
            and self.samples >= self.min_samples
            and self.failure_rate > self.failure_ratio_threshold
        ):
            self.state = CircuitState.OPEN
            self._opened_at = self.clock()
            self.trips += 1

    def snapshot(self) -> dict[str, object]:
        """给 trace / 报表用。"""
        return {
            "name": self.name,
            "state": self.state.value,
            "samples": self.samples,
            "failure_rate": round(self.failure_rate, 4),
            "open_seconds": self.open_seconds,
            "trips": self.trips,
            "rejections": self.rejections,
        }
