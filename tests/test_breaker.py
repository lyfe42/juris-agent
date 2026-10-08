"""熔断器测试。三条断言分别对应三个经典错误实现。"""

from __future__ import annotations

from lexguard.tools.breaker import CircuitBreaker, CircuitState


def make(clock, **kw) -> CircuitBreaker:
    defaults = dict(
        name="t",
        window=10,
        min_samples=5,
        failure_ratio_threshold=0.5,
        open_seconds=30.0,
        half_open_probes=2,
        clock=clock,
    )
    defaults.update(kw)
    return CircuitBreaker(**defaults)


class TestMinSamples:
    """错误实现 A：没有 min_samples，冷启动第一个请求失败就把工具打死。"""

    def test_few_failures_do_not_open(self, clock) -> None:
        b = make(clock)
        for _ in range(4):  # min_samples = 5
            assert b.allow()[0] is True
            b.record(False)
        assert b.state is CircuitState.CLOSED

    def test_opening_at_min_samples(self, clock) -> None:
        b = make(clock)
        for _ in range(5):
            b.allow()
            b.record(False)
        assert b.state is CircuitState.OPEN
        assert b.failure_rate == 1.0

    def test_empty_window_failure_rate_is_zero(self, clock) -> None:
        assert make(clock).failure_rate == 0.0

    def test_ratio_threshold_is_strict(self, clock) -> None:
        """失败率正好等于阈值不熔断（法条式边界思维在这里同样适用）。"""
        b = make(clock, min_samples=4, failure_ratio_threshold=0.5)
        for i in range(4):
            b.allow()
            b.record(i % 2 == 0)  # 2 成功 2 失败 → 0.5，不 > 0.5
        assert b.state is CircuitState.CLOSED


class TestOpenState:
    def test_rejects_while_open(self, clock) -> None:
        b = make(clock, min_samples=2)
        for _ in range(2):
            b.allow()
            b.record(False)
        allowed, wait = b.allow()
        assert allowed is False
        assert 0 < wait <= 30.0

    def test_wait_decreases_with_time(self, clock) -> None:
        b = make(clock, min_samples=2)
        for _ in range(2):
            b.allow()
            b.record(False)
        _, w1 = b.allow()
        clock.advance(10)
        _, w2 = b.allow()
        assert w2 < w1

    def test_transitions_to_half_open_after_cooldown(self, clock) -> None:
        b = make(clock, min_samples=2, half_open_probes=2)
        for _ in range(2):
            b.allow()
            b.record(False)
        clock.advance(31)
        assert b.allow()[0] is True
        assert b.state is CircuitState.HALF_OPEN


class TestHalfOpen:
    """错误实现 B：半开无限放行（等于没熔断）或只放行一个（抖动即重开）。"""

    def test_limits_probes(self, clock) -> None:
        b = make(clock, min_samples=2, half_open_probes=2)
        for _ in range(2):
            b.allow()
            b.record(False)
        clock.advance(31)
        assert b.allow()[0] is True
        assert b.allow()[0] is True
        assert b.allow()[0] is False  # 探测额度用尽

    def test_probe_success_closes_and_clears_window(self, clock) -> None:
        """错误实现 C：不清空窗口 → 下一个请求立刻又熔断，形成抖动循环。"""
        b = make(clock, min_samples=2, half_open_probes=2, window=10)
        for _ in range(2):
            b.allow()
            b.record(False)
        clock.advance(31)
        b.allow()
        b.record(True)
        assert b.state is CircuitState.CLOSED
        assert b.samples == 0
        assert b.failure_rate == 0.0

    def test_probe_failure_reopens(self, clock) -> None:
        b = make(clock, min_samples=2, half_open_probes=2)
        for _ in range(2):
            b.allow()
            b.record(False)
        clock.advance(31)
        b.allow()
        b.record(False)
        assert b.state is CircuitState.OPEN
        # 冷却重新计时：立刻请求应被拒
        assert b.allow()[0] is False

    def test_success_before_trip_keeps_closed(self, clock) -> None:
        b = make(clock, min_samples=4, failure_ratio_threshold=0.5)
        for _ in range(4):
            b.allow()
            b.record(True)
        assert b.state is CircuitState.CLOSED


class TestCounters:
    """trips / rejections 不是装饰：只看成功率无法区分"熔断没生效"和"没装熔断"。"""

    def test_trips_counts_each_open_transition(self, clock) -> None:
        b = make(clock, min_samples=2, half_open_probes=1)
        assert b.trips == 0
        for _ in range(2):
            b.allow()
            b.record(False)
        assert b.trips == 1
        # 半开探测再失败 → 第二次跳闸
        clock.advance(31)
        b.allow()
        b.record(False)
        assert b.trips == 2

    def test_rejections_count_refused_calls(self, clock) -> None:
        b = make(clock, min_samples=2)
        for _ in range(2):
            b.allow()
            b.record(False)
        for _ in range(3):
            assert b.allow()[0] is False
        assert b.rejections == 3

    def test_counters_survive_window_clearing(self, clock) -> None:
        """窗口被半开探测清空后计数器要保留，否则报表里熔断"从没发生过"。"""
        b = make(clock, min_samples=2, half_open_probes=1)
        for _ in range(2):
            b.allow()
            b.record(False)
        clock.advance(31)
        b.allow()
        b.record(True)
        assert b.samples == 0
        assert b.trips == 1

    def test_healthy_breaker_never_trips(self, clock) -> None:
        """场景 A 的断言：均匀低幅度错误不该触发熔断。"""
        b = make(clock)
        for _ in range(300):
            b.allow()
            b.record(True)
        assert b.trips == 0
        assert b.rejections == 0


class TestSnapshot:
    def test_snapshot_shape(self, clock) -> None:
        b = make(clock)
        b.allow()
        b.record(False)
        s = b.snapshot()
        assert s["state"] == "closed"
        assert s["samples"] == 1
        assert "failure_rate" in s
        assert s["trips"] == 0
        assert s["rejections"] == 0
