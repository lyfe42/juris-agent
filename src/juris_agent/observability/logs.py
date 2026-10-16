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


def configure_logging(
    *, level: str = "INFO", json: bool = False, stream: object = None
) -> None:
    """配一次，全进程生效。**可以重复调用。**

    `json=True` 用于生产（丢给日志平台），`json=False` 用于终端。
    判据由调用方给（API 的 lifespan 会读 `settings.is_prod`），
    这个函数不猜——库不该替应用决定"现在算不算生产"。

    ## `stream` 为什么存在（默认 `None` = 保持原样：日志进 stdout）

    服务端和命令行的诉求**正好相反**：

    - **服务端**：日志和业务输出都进 stdout，由容器/日志平台统一收走。
      这是默认值，也是原来唯一的行为。
    - **命令行**：stdout 得**干干净净只有答案**。因为 `juris ask "…" > answer.txt`
      和 `juris ask --json "…" | jq` 都是真实用法，
      而日志（我们自己的 + httpx 那种第三方的，见 `cli.py`）一旦掺进去，
      存下来的文件里就混着十几行时间戳，管道那头直接解析失败。
      **它破坏的是 `cli.py` 模块 docstring 里承诺的"输出分流"，
      而破坏发生在别处**——从 `cli.py` 里根本看不出来。

    ⚠️ 默认值保持 `stdout` 是**刻意的**：改默认值会让服务端的日志
    静默地换一个流，而那是在容器里排查问题时才会发现的事。
    要 stderr 的调用方显式传，代价是多打四个字。

    ⚠️ **`sys.stderr` 这类东西要在调用点时求值，不能写成默认参数**
    （`stream=sys.stderr` 会在**定义时**绑定，那时 pytest 的
    capture 还没装上，结果是日志绕过了捕获、打到真终端上）。

    ⚠️ **不传 `stream` 和"传一个 stdout"不是一回事**，这个区别很要命，
    下面是实测踩出来的：

    | 传什么 | structlog 拿到的 | 后果 |
    |---|---|---|
    | 不传（`None`） | `PrintLogger(file=None)` | 每次打印**现取** `sys.stdout`，跟着替换走 |
    | 传一个具体对象 | `PrintLogger(file=那个对象)` | **永久绑死在那个对象上** |

    第一版我写的是"不传就当 `sys.stdout`"——也就是两种情况都绑死。
    结果是**整套测试红了 13 条**，报的还是
    `ValueError: I/O operation on closed file`：

        某个用 `TestClient` 的测试走 API 的 lifespan → `configure_logging()`
        → 绑住了 **pytest 当时的捕获对象** → 那个测试一结束、捕获对象被关掉
        → **从那一刻起，后面每一条打日志的测试都炸**，
          而报错位置全在**跟日志无关**的代码里。

    **这个坑在生产里同样成立**（只是不会炸）：绑死意味着
    "以后换 `sys.stdout`（重定向、被别的框架接管）日志还往老地方写"。
    所以默认这一支必须保持 structlog 的"现取"行为——**那是原来的行为**，
    也是唯一不会在别人手里埋雷的行为。要绑死只能由调用方显式要求。
    """
    global _CONFIGURED

    effective_stream = sys.stdout if stream is None else stream

    # 标准库的根日志器也要设：uvicorn、httpx 这些第三方库走的是标准库，
    # 不设的话它们会在我们精心格式化过的日志中间插入一行丑陋的默认格式。
    #
    # ⚠️ `basicConfig` 在**根日志器已经有 handler 时是空操作**。
    # 所以如果某个第三方库先配过日志，这里设的 stream 不会生效——
    # 不传 `force=True` 去强拆，是因为那会连带把 pytest 的捕获 handler
    # 一起拆掉（测试里日志就变成打到真终端上的了）。
    # 代价是"日志没进 stderr"这种偏差可能悄悄发生，所以在这里记一笔。
    logging.basicConfig(format="%(message)s", stream=effective_stream, level=level.upper())

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
        else structlog.dev.ConsoleRenderer(colors=_isatty(effective_stream))
    )
    # 见上面那张表：`stream is None` 时**故意不传 `file`**，让 structlog
    # 每次打印现取 `sys.stdout`。这不是"少写一个参数"，是两种行为。
    #
    # ⚠️ `PrintLoggerFactory` 的形参名是 `file`，不是 `stream`。
    # 写成 `stream=stream` 会得到 `TypeError`，而且是在
    # "日志第一次真的被写出去"的那一刻才炸——比配置时炸难查得多。
    factory = (
        structlog.PrintLoggerFactory()
        if stream is None
        else structlog.PrintLoggerFactory(file=stream)
    )
    structlog.configure(
        processors=[*processors, renderer],  # type: ignore[list-item]
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=factory,
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True


def _isatty(stream: object) -> bool:
    """这个流是不是终端。**拿不到就当不是。**

    `ConsoleRenderer(colors=True)` 往一个不是终端的地方（文件、管道、
    IDE 的输出面板）写 ANSI 转义序列，结果是一堆 `[32m` 之类的乱码——
    而且**只在重定向的时候出现**，直接在终端里跑是好的，
    很容易被误判成"玄学"。判不出来时按"不是终端"处理，
    宁可少点颜色，也不要往文件里写转义字符。
    """
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except (ValueError, OSError):
        # 流已经关了的时候 `isatty()` 会抛。同上：当成"不是终端"。
        return False


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
