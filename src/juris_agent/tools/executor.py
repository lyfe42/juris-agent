"""工具执行层 —— 全项目的核心，也是面试追问最密集的地方。

一次 `ToolExecutor.call()` 的完整流水线：

```
幂等键命中? ──是──> 直接返回缓存结果（不消耗任何重试预算）
     │否
入参 schema 校验 ──失败──> 返回字段级错误，**不重试**（同样入参必然同样失败）
     │通过
熔断器放行? ──否──> 返回 CircuitOpenError（压根不发起调用，保护下游）
     │是
┌─> 调用（带超时）
│     失败且可重试且还有预算 ──> 指数退避(±20% 抖动) ──┐
└────────────────────────────────────────────────────┘
     │成功                          │重试耗尽
     │                              │
出参 schema 校验                    降级实现（若有）──> status=degraded
     │                              │无降级
结果写入幂等缓存 ──> status=ok       status=failed（附带结构化建议）
```

**几个刻意的设计决定，每一条都能被追问：**

- **失败的返回值是结构化的，不是空字符串**。模型拿到 `{"error_type","hint"}` 才有
  机会换工具或改参数；拿到空串只能原地重试。
- **被熔断拒绝的调用不计入熔断窗口**。否则拒绝次数会自我强化，熔断永不解开。
- **熔断窗口的单位是"一次逻辑调用"而不是"一次物理尝试"**：重试 3 次后成功算 1 次
  成功。理由是这反映的是"用户请求这个工具时失败的概率"，而不是下游的瞬时错误率。
  两者都需要，所以 `ToolOutcome.attempts` 单独记录物理尝试次数供报表用。
- **入参校验失败不计入熔断**：那是模型的错，不是工具的错。把它算进工具失败率会
  让工具被模型的一句话冤枉掉。
- **时钟、随机数、sleep 全部可注入**。故障注入实验要跑 300 次调用并复现同样的
  抖动序列，测试更是一次都不能真 sleep。
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from juris_agent.config import Settings, ToolSettings
from juris_agent.tools.breaker import CircuitBreaker
from juris_agent.tools.errors import (
    CircuitOpenError,
    PermanentToolError,
    RetryPolicy,
    ToolError,
    ToolTimeoutError,
    ToolValidationError,
    is_retryable,
)
from juris_agent.tools.registry import ToolRegistry, ToolSpec

# 同步工具的超时靠线程池实现。被超时放弃的线程**无法强制中断**（Python 没有
# 安全的线程终止），它会继续跑完——这是 Python 的固有限制，不是这里的 bug。
# 对本项目的工具（纯计算 + 一次 HTTP 调用，都带自己的 httpx 超时）代价可接受。
# 真需要硬超时要用子进程，成本远大于收益。
_TIMEOUT_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="juris-agent-tool")


class ToolStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    """走了降级路径。结果可用但精度/覆盖度下降。"""
    FAILED = "failed"


class ToolOutcome(BaseModel):
    """工具调用结果。**任何工具失败都通过本对象返回，不向调用方抛异常**。

    这是刻意的：Agent 主循环绝不能因为一个工具挂了就整个崩掉。不可恢复的
    编程错误（工具未注册）才会抛。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: str
    status: ToolStatus
    value: Any = None
    """校验通过的 result_model 实例（或降级实现返回的同类对象）。"""
    attempts: int = 0
    """物理尝试次数（含重试）。故障注入报表的"重试放大延迟"就取自它。"""
    elapsed_ms: float = 0.0
    idempotency_hit: bool = False
    degraded: bool = False
    error: dict[str, Any] | None = None
    breaker_state: str = "closed"
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (ToolStatus.OK, ToolStatus.DEGRADED)

    def payload(self) -> Any:
        """可 JSON 序列化的结果，用于 SSE / record-replay / 喂回模型。"""
        v = self.value
        if isinstance(v, BaseModel):
            return v.model_dump(mode="json")
        return v


def _default_sleeper(seconds: float) -> None:
    time.sleep(seconds)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        settings: Settings | ToolSettings | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = _default_sleeper,
        rng: random.Random | None = None,
        breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        self.registry = registry
        if isinstance(settings, Settings):
            self.settings = settings.tool
        else:
            self.settings = settings or ToolSettings()
        self._clock = clock
        self._sleep = sleeper
        self._rng = rng or random.Random()

        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = breakers or {}
        self._idem: dict[str, tuple[float, ToolOutcome]] = {}

    # —— 熔断器 ——
    def breaker(self, name: str) -> CircuitBreaker:
        with self._lock:
            b = self._breakers.get(name)
            if b is None:
                b = CircuitBreaker(
                    name=name,
                    window=self.settings.breaker_window,
                    min_samples=self.settings.breaker_min_samples,
                    failure_ratio_threshold=self.settings.breaker_failure_ratio,
                    open_seconds=self.settings.breaker_open_seconds,
                    half_open_probes=self.settings.breaker_half_open_probes,
                    clock=self._clock,
                )
                self._breakers[name] = b
            return b

    def breaker_snapshots(self) -> list[dict[str, object]]:
        return [self.breaker(n).snapshot() for n in self.registry.names()]

    # —— 幂等缓存 ——
    def _idem_get(self, key: str) -> ToolOutcome | None:
        with self._lock:
            entry = self._idem.get(key)
            if entry is None:
                return None
            stored_at, outcome = entry
            if self._clock() - stored_at > self.settings.idempotency_ttl_seconds:
                del self._idem[key]
                return None
            return outcome

    def _idem_put(self, key: str, outcome: ToolOutcome) -> None:
        with self._lock:
            if len(self._idem) >= self.settings.idempotency_max_entries:
                # 简单的过期清理；不引入 LRU 依赖，够用即可
                now = self._clock()
                ttl = self.settings.idempotency_ttl_seconds
                for k in [k for k, (t, _) in self._idem.items() if now - t > ttl]:
                    del self._idem[k]
                if len(self._idem) >= self.settings.idempotency_max_entries:
                    oldest = min(self._idem, key=lambda k: self._idem[k][0])
                    del self._idem[oldest]
            self._idem[key] = (self._clock(), outcome)

    # —— 退避 ——
    def backoff_seconds(self, attempt: int) -> float:
        """指数退避 + ±jitter 抖动。attempt 从 1 开始（第 1 次失败后的等待）。

        抖动是必需的而非锦上添花：没有抖动时，一批并发请求会同时重试，
        形成对下游的周期性脉冲（thundering herd），把正在恢复的服务再打垮。
        """
        base = self.settings.backoff_base_ms / 1000.0
        raw = base * (2 ** (attempt - 1))
        jitter = self._rng.uniform(-1.0, 1.0) * self.settings.backoff_jitter
        return min(raw * (1.0 + jitter), self.settings.backoff_max_ms / 1000.0)

    # —— 调用 ——
    def call(
        self,
        name: str,
        raw_args: dict[str, Any] | BaseModel | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> ToolOutcome:
        t0 = self._clock()
        started_ms = t0 * 1000.0

        def elapsed() -> float:
            return self._clock() * 1000.0 - started_ms

        spec = self.registry.get(name)  # 未注册 = 编程错误，抛出
        breaker = self.breaker(name)

        # ① 幂等
        if spec.idempotent and idempotency_key:
            hit = self._idem_get(f"{name}:{idempotency_key}")
            if hit is not None:
                return hit.model_copy(
                    update={"idempotency_hit": True, "elapsed_ms": elapsed()}
                )

        # ② 入参校验（失败不重试、不计熔断）
        try:
            args = self._validate_args(spec, raw_args)
        except ToolValidationError as e:
            return ToolOutcome(
                tool=name,
                status=ToolStatus.FAILED,
                attempts=0,
                elapsed_ms=elapsed(),
                error=e.to_feedback(),
                breaker_state=breaker.state.value,
            )

        # ③ 熔断闸门
        allowed, wait_s = breaker.allow()
        if not allowed:
            err = CircuitOpenError(
                f"工具 {name} 处于熔断状态，已拒绝调用", tool=name, retry_after_s=wait_s
            )
            return ToolOutcome(
                tool=name,
                status=ToolStatus.FAILED,
                attempts=0,
                elapsed_ms=elapsed(),
                error=err.to_feedback(),
                breaker_state=breaker.state.value,
            )

        # ④ 重试循环
        max_attempts = max(1, self.settings.max_retries)
        last_error: BaseException | None = None
        attempts = 0

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                raw_result = self._invoke(spec, args)
            except BaseException as exc:  # noqa: BLE001 - 统一分类后决定是否重试
                last_error = self._classify(exc, name)
                if not is_retryable(last_error) or attempt == max_attempts:
                    break
                self._sleep(self.backoff_seconds(attempt))
                continue

            # 出参校验：不过就是工具的 bug，不重试
            try:
                value = self._validate_result(spec, raw_result)
            except ToolValidationError as exc:
                last_error = exc
                break

            outcome = ToolOutcome(
                tool=name,
                status=ToolStatus.OK,
                value=value,
                attempts=attempts,
                elapsed_ms=elapsed(),
                breaker_state="closed",
            )
            breaker.record(True)
            outcome.breaker_state = breaker.state.value
            if spec.idempotent and idempotency_key:
                self._idem_put(f"{name}:{idempotency_key}", outcome)
            return outcome

        # ⑤ 全部失败 → 记一次调用失败
        breaker.record(False)
        assert last_error is not None

        # ⑥ 降级
        if spec.fallback is not None:
            try:
                fb_raw = spec.fallback(args, last_error)
                fb_value = self._validate_result(spec, fb_raw)
                return ToolOutcome(
                    tool=name,
                    status=ToolStatus.DEGRADED,
                    value=fb_value,
                    attempts=attempts,
                    elapsed_ms=elapsed(),
                    degraded=True,
                    error=self._feedback(last_error, name),
                    breaker_state=breaker.state.value,
                    warnings=[
                        f"已降级：{type(last_error).__name__}。结果可用，但精度或覆盖度下降。"
                    ],
                )
            except BaseException as fb_exc:  # noqa: BLE001
                return ToolOutcome(
                    tool=name,
                    status=ToolStatus.FAILED,
                    attempts=attempts,
                    elapsed_ms=elapsed(),
                    error=self._feedback(last_error, name),
                    breaker_state=breaker.state.value,
                    warnings=[f"降级路径同样失败：{type(fb_exc).__name__}: {fb_exc}"],
                )

        return ToolOutcome(
            tool=name,
            status=ToolStatus.FAILED,
            attempts=attempts,
            elapsed_ms=elapsed(),
            error=self._feedback(last_error, name),
            breaker_state=breaker.state.value,
        )

    # —— 内部 ——
    def _validate_args(
        self, spec: ToolSpec, raw: dict[str, Any] | BaseModel | None
    ) -> BaseModel:
        if isinstance(raw, spec.args_model):
            return raw
        payload = raw.model_dump() if isinstance(raw, BaseModel) else (raw or {})
        try:
            return spec.args_model.model_validate(payload)
        except ValidationError as e:
            raise ToolValidationError(
                f"工具 {spec.name} 的入参不符合 schema",
                tool=spec.name,
                direction="input",
                field_errors=[
                    {
                        "loc": ".".join(str(p) for p in err["loc"]),
                        "msg": err["msg"],
                        "type": err["type"],
                    }
                    for err in e.errors()
                ],
            ) from e

    def _validate_result(self, spec: ToolSpec, raw: Any) -> Any:
        if isinstance(raw, spec.result_model):
            return raw
        try:
            return spec.result_model.model_validate(raw)
        except ValidationError as e:
            raise ToolValidationError(
                f"工具 {spec.name} 的返回值不符合 schema（工具实现有 bug）",
                tool=spec.name,
                direction="output",
                field_errors=[
                    {
                        "loc": ".".join(str(p) for p in err["loc"]),
                        "msg": err["msg"],
                        "type": err["type"],
                    }
                    for err in e.errors()
                ],
            ) from e

    def _invoke(self, spec: ToolSpec, args: BaseModel) -> Any:
        timeout = (
            self.settings.call_timeout_seconds
            if spec.timeout_seconds is None
            else spec.timeout_seconds
        )
        if not timeout or timeout <= 0:
            return spec.fn(args)

        future = _TIMEOUT_POOL.submit(spec.fn, args)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as e:
            future.cancel()
            raise ToolTimeoutError(
                f"工具 {spec.name} 超过 {timeout}s 未返回", tool=spec.name, cause=e
            ) from e

    @staticmethod
    def _classify(exc: BaseException, tool: str) -> BaseException:
        """把任意异常归入 ToolError 体系。

        未归类的异常按 **不可重试** 处理：重试一个 `TypeError` 不会改变结果。
        这一条是刻意保守的——把未知错误当成可重试，会掩盖真正的 bug。
        """
        if isinstance(exc, ToolError):
            if exc.tool is None:
                exc.tool = tool
            return exc
        return PermanentToolError(f"{type(exc).__name__}: {exc}", tool=tool, cause=exc)

    @staticmethod
    def _feedback(exc: BaseException, tool: str) -> dict[str, Any]:
        if isinstance(exc, ToolError):
            return exc.to_feedback()
        return {
            "error_type": type(exc).__name__,
            "retry_policy": RetryPolicy.PERMANENT.value,
            "message": str(exc),
            "tool": tool,
        }
