"""工具层的错误分类。

**这个模块是整个执行层的语义基础**：重试/熔断/降级三种机制全都只看
`RetryPolicy` 一个字段。把"错在哪一类"和"该怎么应对"绑在一起，是为了避免
散落各处的 `if status == 429` 式判断——那种代码的必然结局是某处漏了一个分支。

分类依据：**重试是否可能改变结果**。

- `TRANSIENT`：网络抖动、超时、5xx、限流。重试有意义。
- `PERMANENT`：4xx 参数/鉴权/额度类错误，以及工具内部逻辑错误。**重试只会浪费
  时间和额度**——同样的输入必然得到同样的结果。这是最常被写错的一类：
  把 401/403/404 也塞进重试循环，故障注入实验里表现为"重试了 3 次还是失败，
  但延迟翻了 3 倍"。
- `VALIDATION`：模型给的参数不符合 schema。**也不重试**，而是把字段级错误
  回灌给模型让它改参数——重试同样的参数是确定性的失败。
- `CIRCUIT_OPEN`：熔断器打开，压根没发起调用。
"""

from __future__ import annotations

from enum import Enum


class RetryPolicy(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    VALIDATION = "validation"
    CIRCUIT_OPEN = "circuit_open"


class ToolError(Exception):
    """工具层异常基类。`policy` 决定执行层怎么应对。"""

    policy: RetryPolicy = RetryPolicy.PERMANENT

    def __init__(
        self,
        message: str,
        *,
        tool: str | None = None,
        policy: RetryPolicy | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.tool = tool
        if policy is not None:
            self.policy = policy
        self.cause = cause

    def to_feedback(self) -> dict[str, object]:
        """给模型看的结构化反馈。

        工具失败时返回"空结果"是错的做法——模型不知道发生了什么，只能原地重试。
        返回错误类型 + 可操作建议，模型才有机会换工具或换参数。
        """
        return {
            "error_type": type(self).__name__,
            "retry_policy": self.policy.value,
            "message": str(self),
            "tool": self.tool,
        }


class TransientToolError(ToolError):
    """可重试：网络/超时/5xx/限流。"""

    policy = RetryPolicy.TRANSIENT


class PermanentToolError(ToolError):
    """不可重试：4xx、额度耗尽、鉴权失败、工具内部逻辑错误。"""

    policy = RetryPolicy.PERMANENT


class ToolValidationError(ToolError):
    """入参或出参不符合 schema。携带字段级错误供回灌。"""

    policy = RetryPolicy.VALIDATION

    def __init__(
        self,
        message: str,
        *,
        tool: str | None = None,
        direction: str = "input",
        field_errors: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(message, tool=tool)
        self.direction = direction  # "input" | "output"
        self.field_errors = field_errors or []

    def to_feedback(self) -> dict[str, object]:
        fb = super().to_feedback()
        fb["direction"] = self.direction
        fb["field_errors"] = self.field_errors
        fb["hint"] = (
            "请修正上述字段后重新调用。常见原因：类型不符（字符串 vs 数值）、"
            "缺少必填字段、枚举取值不在允许范围内。"
        )
        return fb


class CircuitOpenError(ToolError):
    """熔断器打开，未发起调用。"""

    policy = RetryPolicy.CIRCUIT_OPEN

    def __init__(self, message: str, *, tool: str | None = None, retry_after_s: float = 0.0):
        super().__init__(message, tool=tool)
        self.retry_after_s = retry_after_s

    def to_feedback(self) -> dict[str, object]:
        fb = super().to_feedback()
        fb["retry_after_seconds"] = round(self.retry_after_s, 1)
        fb["hint"] = (
            "该工具连续失败已被熔断。请改用其他工具，或基于已有信息作答，"
            "不要重复调用本工具。"
        )
        return fb


class ToolTimeoutError(TransientToolError):
    """调用超时。按可重试处理，但连续超时会触发熔断。"""


def is_retryable(err: BaseException) -> bool:
    return getattr(err, "policy", RetryPolicy.PERMANENT) is RetryPolicy.TRANSIENT


def classify_http_status(status_code: int) -> RetryPolicy:
    """把 HTTP 状态码映射成重试策略。

    408/429 与 5xx 可重试；其余 4xx 不可重试。这个划分要能背下来：
    面试里"哪些错误不该重试"几乎必问。
    """
    if status_code in (408, 425, 429):
        return RetryPolicy.TRANSIENT
    if 500 <= status_code < 600:
        return RetryPolicy.TRANSIENT
    if 400 <= status_code < 500:
        return RetryPolicy.PERMANENT
    return RetryPolicy.PERMANENT
