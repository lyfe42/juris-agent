"""工具执行层测试 —— 项目里最该被测厚的一层。

每个测试对应执行层流水线里的一个决策点，命名尽量直白地说出"这条保证的是什么"。
所有测试用注入的假时钟与假 sleeper，所以整套跑在毫秒级且完全可复现。
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel, Field

from lexguard.config import ToolSettings
from lexguard.tools.errors import (
    PermanentToolError,
    TransientToolError,
    classify_http_status,
)
from lexguard.tools.executor import ToolExecutor, ToolStatus
from lexguard.tools.registry import ToolRegistry, ToolSpec


class PingArgs(BaseModel):
    n: int = Field(ge=0)


class PingResult(BaseModel):
    n: int
    tag: str = "pong"


def build(fn, *, fallback=None, timeout: float = 0.0, idempotent: bool = True, **kw):
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="ping",
            description="test tool",
            args_model=PingArgs,
            result_model=PingResult,
            fn=fn,
            fallback=fallback,
            idempotent=idempotent,
            timeout_seconds=timeout,
        )
    )
    settings = ToolSettings(**kw)
    return reg, settings


def executor_for(fn, *, clock, sleeper, **kw) -> ToolExecutor:
    reg, settings = build(fn, **kw)
    return ToolExecutor(reg, settings, clock=clock, sleeper=sleeper)


class TestHappyPath:
    def test_ok(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        out = ex.call("ping", {"n": 7})
        assert out.status is ToolStatus.OK
        assert out.attempts == 1
        assert out.value.n == 7
        assert out.ok is True
        assert sleeper.calls == []

    def test_accepts_model_input(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        out = ex.call("ping", PingArgs(n=3))
        assert out.value.n == 3

    def test_payload_is_json_serializable(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        assert ex.call("ping", {"n": 1}).payload() == {"n": 1, "tag": "pong"}


class TestInputValidation:
    def test_bad_input_fails_without_retrying(self, clock, sleeper) -> None:
        calls = []
        ex = executor_for(
            lambda a: calls.append(1) or PingResult(n=a.n), clock=clock, sleeper=sleeper
        )
        out = ex.call("ping", {"n": -5})
        assert out.status is ToolStatus.FAILED
        assert out.attempts == 0          # 没发起调用
        assert calls == []                # 工具函数压根没被执行
        assert out.error["direction"] == "input"
        assert out.error["field_errors"][0]["loc"] == "n"

    def test_missing_required_field_reports_loc(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        out = ex.call("ping", {})
        assert out.error["field_errors"][0]["loc"] == "n"

    def test_type_coercion_failure_is_reported_not_raised(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        out = ex.call("ping", {"n": "not-a-number"})
        assert out.status is ToolStatus.FAILED
        assert out.error["retry_policy"] == "validation"

    def test_validation_does_not_count_against_breaker(self, clock, sleeper) -> None:
        """入参校验失败是模型的错，不该把工具熔断掉。"""
        ex = executor_for(
            lambda a: PingResult(n=a.n),
            clock=clock,
            sleeper=sleeper,
            breaker_min_samples=3,
        )
        for _ in range(10):
            ex.call("ping", {"n": -1})
        assert ex.breaker("ping").state.value == "closed"
        assert ex.call("ping", {"n": 1}).ok is True

    def test_unregistered_tool_raises(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        with pytest.raises(KeyError):
            ex.call("nope", {})


class TestRetry:
    def test_transient_is_retried_then_gives_up(self, clock, sleeper) -> None:
        def boom(a):
            raise TransientToolError("network down")

        ex = executor_for(boom, clock=clock, sleeper=sleeper, max_retries=3)
        out = ex.call("ping", {"n": 1})
        assert out.status is ToolStatus.FAILED
        assert out.attempts == 3
        assert len(sleeper.calls) == 2   # 3 次尝试之间只 sleep 2 次

    def test_transient_then_success(self, clock, sleeper) -> None:
        state = {"calls": 0}

        def flaky(a):
            state["calls"] += 1
            if state["calls"] == 1:
                raise TransientToolError("blip")
            return PingResult(n=a.n)

        ex = executor_for(flaky, clock=clock, sleeper=sleeper, max_retries=3)
        out = ex.call("ping", {"n": 2})
        assert out.status is ToolStatus.OK
        assert out.attempts == 2
        assert out.value.n == 2

    def test_permanent_is_not_retried(self, clock, sleeper) -> None:
        def boom(a):
            raise PermanentToolError("401 unauthorized")

        ex = executor_for(boom, clock=clock, sleeper=sleeper, max_retries=3)
        out = ex.call("ping", {"n": 1})
        assert out.attempts == 1
        assert sleeper.calls == []
        assert out.error["retry_policy"] == "permanent"

    def test_unknown_exception_is_treated_as_permanent(self, clock, sleeper) -> None:
        """保守选择：未知错误不重试，否则会掩盖真正的 bug。"""

        def boom(a):
            raise TypeError("a bug in the tool")

        ex = executor_for(boom, clock=clock, sleeper=sleeper, max_retries=3)
        out = ex.call("ping", {"n": 1})
        assert out.attempts == 1
        assert out.error["retry_policy"] == "permanent"

    def test_backoff_grows_exponentially_within_jitter_bounds(
        self, clock, sleeper
    ) -> None:
        ex = executor_for(
            lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper, backoff_base_ms=200
        )
        b1 = ex.backoff_seconds(1)
        b3 = ex.backoff_seconds(3)
        assert 0.16 <= b1 <= 0.24      # 200ms ±20%
        assert 0.64 <= b3 <= 0.96      # 800ms ±20%

    def test_backoff_is_capped(self, clock, sleeper) -> None:
        ex = executor_for(
            lambda a: PingResult(n=a.n),
            clock=clock,
            sleeper=sleeper,
            backoff_base_ms=200,
            backoff_max_ms=1000,
        )
        assert ex.backoff_seconds(10) <= 1.0

    def test_backoff_actually_has_jitter(self, clock, sleeper) -> None:
        """没有抖动时并发重试会形成对下游的脉冲（thundering herd）。"""
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        samples = {round(ex.backoff_seconds(1), 6) for _ in range(20)}
        assert len(samples) > 1


class TestOutputValidation:
    def test_bad_output_is_a_tool_bug(self, clock, sleeper) -> None:
        def bad(a):
            return {"n": "not-an-int"}

        ex = executor_for(bad, clock=clock, sleeper=sleeper, max_retries=3)
        out = ex.call("ping", {"n": 1})
        assert out.status is ToolStatus.FAILED
        assert out.attempts == 1          # 出参错是确定性的，不重试
        assert out.error["direction"] == "output"

    def test_dict_output_is_coerced(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: {"n": a.n}, clock=clock, sleeper=sleeper)
        out = ex.call("ping", {"n": 4})
        assert isinstance(out.value, PingResult)
        assert out.value.tag == "pong"


class TestCircuitBreakerIntegration:
    def test_opens_and_short_circuits(self, clock, sleeper) -> None:
        calls = []

        def boom(a):
            calls.append(1)
            raise TransientToolError("down")

        ex = executor_for(
            boom,
            clock=clock,
            sleeper=sleeper,
            max_retries=1,
            breaker_min_samples=3,
            breaker_failure_ratio=0.5,
        )
        for _ in range(3):
            assert ex.call("ping", {"n": 1}).status is ToolStatus.FAILED
        assert ex.breaker("ping").state.value == "open"

        before = len(calls)
        out = ex.call("ping", {"n": 1})
        assert out.status is ToolStatus.FAILED
        assert out.attempts == 0                       # 压根没发起调用
        assert len(calls) == before                    # 下游被保护
        assert out.error["retry_policy"] == "circuit_open"
        assert "retry_after_seconds" in out.error

    def test_recovers_after_cooldown(self, clock, sleeper) -> None:
        state = {"fail": True}

        def flaky(a):
            if state["fail"]:
                raise TransientToolError("down")
            return PingResult(n=a.n)

        ex = executor_for(
            flaky,
            clock=clock,
            sleeper=sleeper,
            max_retries=1,
            breaker_min_samples=3,
            breaker_open_seconds=30.0,
        )
        for _ in range(3):
            ex.call("ping", {"n": 1})
        assert ex.call("ping", {"n": 1}).error["retry_policy"] == "circuit_open"

        state["fail"] = False
        clock.advance(31)
        out = ex.call("ping", {"n": 1})
        assert out.status is ToolStatus.OK
        assert ex.breaker("ping").state.value == "closed"

    def test_rejected_calls_do_not_extend_the_window(self, clock, sleeper) -> None:
        """被熔断拒绝的调用不能记进窗口，否则拒绝自我强化、熔断永不解开。"""

        def boom(a):
            raise TransientToolError("down")

        ex = executor_for(
            boom, clock=clock, sleeper=sleeper, max_retries=1, breaker_min_samples=3
        )
        for _ in range(3):
            ex.call("ping", {"n": 1})
        samples_before = ex.breaker("ping").samples
        for _ in range(10):
            ex.call("ping", {"n": 1})
        assert ex.breaker("ping").samples == samples_before


class TestIdempotency:
    def test_same_key_hits_cache(self, clock, sleeper) -> None:
        calls = []

        def fn(a):
            calls.append(1)
            return PingResult(n=a.n)

        ex = executor_for(fn, clock=clock, sleeper=sleeper)
        first = ex.call("ping", {"n": 1}, idempotency_key="k1")
        second = ex.call("ping", {"n": 1}, idempotency_key="k1")

        assert len(calls) == 1
        assert second.idempotency_hit is True
        assert second.value.n == first.value.n

    def test_different_keys_do_not_collide(self, clock, sleeper) -> None:
        calls = []
        ex = executor_for(
            lambda a: calls.append(1) or PingResult(n=a.n), clock=clock, sleeper=sleeper
        )
        ex.call("ping", {"n": 1}, idempotency_key="k1")
        ex.call("ping", {"n": 2}, idempotency_key="k2")
        assert len(calls) == 2

    def test_cache_expires(self, clock, sleeper) -> None:
        calls = []
        ex = executor_for(
            lambda a: calls.append(1) or PingResult(n=a.n),
            clock=clock,
            sleeper=sleeper,
            idempotency_ttl_seconds=10.0,
        )
        ex.call("ping", {"n": 1}, idempotency_key="k1")
        clock.advance(11)
        out = ex.call("ping", {"n": 1}, idempotency_key="k1")
        assert out.idempotency_hit is False
        assert len(calls) == 2

    def test_no_key_means_no_cache(self, clock, sleeper) -> None:
        calls = []
        ex = executor_for(
            lambda a: calls.append(1) or PingResult(n=a.n), clock=clock, sleeper=sleeper
        )
        ex.call("ping", {"n": 1})
        ex.call("ping", {"n": 1})
        assert len(calls) == 2


class TestDegradation:
    def test_fallback_produces_degraded_outcome(self, clock, sleeper) -> None:
        def boom(a):
            raise TransientToolError("down")

        ex = executor_for(
            boom,
            clock=clock,
            sleeper=sleeper,
            max_retries=2,
            fallback=lambda a, err: PingResult(n=a.n, tag="degraded"),
        )
        out = ex.call("ping", {"n": 5})
        assert out.status is ToolStatus.DEGRADED
        assert out.degraded is True
        assert out.ok is True           # 降级结果仍然可用
        assert out.value.tag == "degraded"
        assert out.warnings

    def test_failing_fallback_reports_both_errors(self, clock, sleeper) -> None:
        def boom(a):
            raise TransientToolError("down")

        def bad_fallback(a, err):
            raise RuntimeError("fallback also broken")

        ex = executor_for(
            boom, clock=clock, sleeper=sleeper, max_retries=1, fallback=bad_fallback
        )
        out = ex.call("ping", {"n": 1})
        assert out.status is ToolStatus.FAILED
        assert any("降级路径同样失败" in w for w in out.warnings)

    def test_no_fallback_means_plain_failure(self, clock, sleeper) -> None:
        def boom(a):
            raise TransientToolError("down")

        ex = executor_for(boom, clock=clock, sleeper=sleeper, max_retries=1)
        assert ex.call("ping", {"n": 1}).status is ToolStatus.FAILED


class TestTimeout:
    def test_slow_tool_times_out_as_transient(self, clock, sleeper) -> None:
        def slow(a):
            time.sleep(0.3)
            return PingResult(n=a.n)

        ex = executor_for(slow, clock=clock, sleeper=sleeper, max_retries=1)
        ex.settings.call_timeout_seconds = 0.05
        # spec.timeout_seconds=0.0 表示不套超时，这里显式改成用全局默认
        ex.registry.get("ping").timeout_seconds = None

        out = ex.call("ping", {"n": 1})
        assert out.status is ToolStatus.FAILED
        assert out.error["error_type"] == "ToolTimeoutError"
        assert out.error["retry_policy"] == "transient"


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (408, "transient"),
            (429, "transient"),
            (500, "transient"),
            (503, "transient"),
            (400, "permanent"),
            (401, "permanent"),
            (403, "permanent"),
            (404, "permanent"),
            (422, "permanent"),
        ],
    )
    def test_http_status_mapping(self, status: int, expected: str) -> None:
        assert classify_http_status(status).value == expected


class TestBreakerSnapshots:
    def test_snapshot_covers_all_tools(self, clock, sleeper) -> None:
        ex = executor_for(lambda a: PingResult(n=a.n), clock=clock, sleeper=sleeper)
        ex.call("ping", {"n": 1})
        snaps = ex.breaker_snapshots()
        assert snaps and snaps[0]["name"] == "ping"
        assert snaps[0]["samples"] == 1
