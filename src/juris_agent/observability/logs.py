"""日志的装配（F28）。用 `structlog`，不用标准库 `logging` 直接打字符串。

## 为什么不是 `print()` 或 `logging.info(f"...")`

两个理由，都是排查时才会疼的：

1. **要能按字段筛选。** `logging.info(f"检索完成 {n} 条")` 打出来的是一句话，
   你没法问「所有 `n=0` 的请求是哪几个」——只能拿正则去扒字符串。
   结构化日志是 `log.info("检索完成", n=0, query=...)`，
   `n=0` 是一个**字段**，能筛、能聚合、能被下游的日志系统当列来用。

2. **`request_id` 要自动带上。** 靠 `contextvars` 在处理器里注入
   （见 `context.py`），调用方**一个参数都不用传**。
   靠人肉在每个 `log.info(...)` 里补 `request_id` 的结果必然是"漏了一半"。

## 三个刻意的选择

**① 开发用彩色、生产用 JSON。** 生产要的是"能被机器解析"（丢给日志平台），
开发要的是"人一眼看懂"。用一套格式同时满足两边是做不到的——
JSON 在终端里读起来是一团，彩色在日志平台上会被当成纯文本。

**② `configure_logging()` 可以重复调用，第二次是空操作。**
它会被 CLI 脚本、API 的 lifespan、测试各调一次。
"重复配置日志"在标准库 `logging` 里是经典的"处理器装了两遍、
日志打了两遍"的来源。这里用一个模块级开关挡住。

**③ 库里只提供 `configure`，不自动执行。**
`import juris_agent.observability.logs` 本身**不产生任何副作用**——
不然 `import` 一个模块就会改掉整个进程的日志配置，
而"为什么我的日志格式被改了"这种问题极难查。
"""

from __future__ import annotations

import logging
import sys

import structlog

from juris_agent.observability.context import log_context

# 模块级开关：配置过就不再配。见上面第 ② 条。
_CONFIGURED = False


def _inject_context(
    _logger: object, _method: str, event_dict: dict[str, object]
) -> dict[str, object]:
    """把当前请求的 id 塞进每一条日志。

    **放在处理器链最前面**：后面的处理器（尤其是 JSON 渲染器）
    会把它当成一个普通字段一起序列化出去。

    调用方的字段**优先**：如果某处显式写了 `log.info(msg, request_id="x")`，
    用它的（那是"我知道自己在干什么"的场景，比如补录一条别的请求的日志）。
    """
    ctx = log_context()
    for k, v in ctx.items():
        event_dict.setdefault(k, v)
    return event_dict


def configure_logging(*, level: str = "INFO", json: bool = False) -> None:
    """配一次，全进程生效。**可以重复调用。**

    `json=True` 用于生产（丢给日志平台），`json=False` 用于终端。
    判据由调用方给（API 的 lifespan 会读 `settings.is_prod`），
    这个函数不猜——库不该替应用决定"现在算不算生产"。
    """
    global _CONFIGURED

    # 标准库的根日志器也要设：uvicorn、httpx 这些第三方库走的是标准库，
    # 不设的话它们会在我们精心格式化过的日志中间插入一行丑陋的默认格式。
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    processors: list[object] = [
        _inject_context,
        # 加上时间戳和级别。`TimeStamper` 用 ISO8601（`2026-10-15T10:00:01Z`）——
        # 不用相对时间（"+0.03s"）是因为跨进程对不上：
        # API 进程和评测脚本各打各的日志，相对时间没法互相定位。
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[*processors, renderer],  # type: ignore[list-item]
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """拿一个日志器。

    **`configure_logging()` 没被调用过时会自动配一次**（彩色、INFO）。
    理由：不做这件事的话，`import` 完直接用 structlog 会打到 stderr
    且格式是 structlog 的默认格式，看起来像"日志坏了"。自动配一次
    的结果是"至少能看"，而显式配置过的进程不会被它覆盖（幂等）。
    """
    if not _CONFIGURED:
        configure_logging()
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return logger  # type: ignore[return-value]


__all__ = ["configure_logging", "get_logger"]
