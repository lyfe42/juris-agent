"""一条请求的「身份证」：`request_id`（F28）。

## 要解决的问题

线上出问题时，日志长这样：

```
10:00:01 INFO  检索完成 3 条
10:00:01 INFO  工具调用 calc_severance 失败
10:00:01 INFO  检索完成 8 条
10:00:02 INFO  工具调用 calc_severance 成功
```

**哪两行属于同一个用户？** 看不出来。并发一上来，几十条请求的日志交织在一起，
而你手上只有「某位用户说答案不对」这一条线索。

解法是给每条请求发一个 id，写进这条请求产生的**每一行日志**里。
之后 `grep '<id>' app.log` 就能还原这一条请求完整的一生。

## 为什么用 `contextvars` 而不是把它当参数传下去

最笨但对的办法是 `def search(query, request_id)` 一路传——**但那样每个函数签名
都要多一个只管日志的参数**，而且中间隔着十几个库函数（BM25、numpy、state、
工具执行器），根本传不进去。

`contextvars` 是标准库给「这段代码在哪个请求里跑」准备的：`set` 之后，
**同一请求内**的任何代码 `get` 都能读到，跨请求互不干扰。
asyncio 里每个 Task 有自己的一份拷贝，所以并发是安全的——
这一点很重要，本项目是 `async` 的 SSE 服务，不使用 contextvars 的话，
并发请求的 `request_id` 会互相覆盖。

## 一个容易踩的坑：`set` 了不还原

`ContextVar.set()` 返回一个 token，用它才能还原。
**只 `set` 不还原，会把上一个请求的 id 泄漏给下一个请求**——
症状是两条不相干的请求在日志里被串成了一条，比没有 request_id 更误导人。

所以这个模块**只暴露 `bind()` 这个上下文管理器**，不导出 `set`：
用完自动还原，调用方没有机会写错。
"""

from __future__ import annotations

import contextlib
import secrets
from collections.abc import Iterator
from contextvars import ContextVar

# 请求 id：12 个十六进制字符。
# 为什么不用 uuid4 那种 36 字符的——日志每一行都要带它，
# 36 字符 × 每请求几十行是一笔真实的体积；12 个十六进制位（48 bit）
# 在本项目的量级下撞号概率可以忽略，而**肉眼能看全**（不用复制粘贴去比对）。
_REQUEST_ID: ContextVar[str] = ContextVar("juris_request_id", default="")
_SESSION_ID: ContextVar[str] = ContextVar("juris_session_id", default="")


def new_request_id() -> str:
    """造一个新的请求 id。

    用 `secrets` 而不是 `random`：请求 id 会出现在日志和响应里，
    可预测的 id 让「猜别人的 request_id 去翻日志」成为可能
    （虽然本项目的日志不外露，但**默认选安全的那个**比事后补便宜）。
    """
    return secrets.token_hex(6)


@contextlib.contextmanager
def bind(
    *,
    request_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[str]:
    """在这段代码里绑定 `request_id` / `session_id`，退出时自动还原。

    用法：

        with bind(request_id="a1b2c3d4e5f6", session_id="thread-7") as rid:
            log.info("开始处理")     # 这一行里自动带上 rid

    **两个 id 的区别**（它们回答的是两个不同的问题）：

    - `request_id`：**这一次请求**。排查"这条回答了没有"用它。
    - `session_id`：**这一段对话**。排查"用户第 3 轮说的月薪为什么没了"用它——
      那一轮有它自己的 request_id，但一段会话的 request_id 是**不同的几个**，
      只有 session_id 能把它们串起来。

    只传一个也能用：没传的那个保持原值（嵌套绑定常见于
    "外层按会话绑定、内层每轮换 request_id"）。
    """
    rid = request_id if request_id is not None else _REQUEST_ID.get()
    sid = session_id if session_id is not None else _SESSION_ID.get()
    tok_r = _REQUEST_ID.set(rid)
    tok_s = _SESSION_ID.set(sid)
    try:
        yield rid
    finally:
        # 顺序反着还原。`reset` 而不是 `set("")`——reset 才是
        # 「回到我进来之前的样子」，set("") 是「假装什么都没有」，
        # 后者在嵌套场景下会把外层绑定的值一起抹掉。
        _SESSION_ID.reset(tok_s)
        _REQUEST_ID.reset(tok_r)


def current_request_id() -> str:
    """当前请求的 id。**没绑定时返回空串**，不抛异常。

    为什么不抛：日志处理器、成本记账这些地方都会读它，
    而它们在脚本、单测、离线评测里一样会被调用（那时根本没有 HTTP 请求）。
    为一个"只是没有"的值抛异常，会让所有离线调用方都要写一遍 try。
    """
    return _REQUEST_ID.get()


def current_session_id() -> str:
    """当前会话的 id。同上，没绑定时是空串。"""
    return _SESSION_ID.get()


def log_context() -> dict[str, str]:
    """给 structlog 用的上下文字典。空值不进字典。

    **为什么空值要丢掉**：日志里每行都挂一个 `request_id=` 然后值是空的，
    读的人会以为"这里本该有个 id 但丢了"，从而去查一个不存在的 bug。
    没有就不写，是最不会误导人的做法。
    """
    out: dict[str, str] = {}
    if rid := _REQUEST_ID.get():
        out["request_id"] = rid
    if sid := _SESSION_ID.get():
        out["session_id"] = sid
    return out


__all__ = [
    "bind",
    "current_request_id",
    "current_session_id",
    "log_context",
    "new_request_id",
]
