"""HTTP 服务（F29）。`uvicorn juris_agent.api.app:app` 起它。

## 一次流式问答在这条链路上怎么走

```
浏览器 ──POST /api/chat/stream──▶ FastAPI
                                    │
                          ①开一段会话（或拿到已有的）
                          ②拿这把会话的锁（同一个会话串行）
                          ③记下版本号（写回时要校验它没变）
                                    │
                     ┌──────────────┴───────────────┐
                     │  asyncio.to_thread           │   ← 图是同步代码，
                     │    runner.stream(...)        │     丢给工作线程跑
                     │      ↓ 每个节点               │
                     │      ↓ LLM 每吐一个字         │
                     └──────────────┬───────────────┘
                                    │  call_soon_threadsafe（跨线程回事件循环）
                                    ▼
                              一条 asyncio.Queue
                                    │
                                    ▼
                      event: memory / route / retrieval / …
                      event: answer      （逐字）
                      event: done        （含成本、trace、引用）
```

## 为什么图要丢到线程里跑——以及为什么这不是"多余的一层"

`AgentRunner.stream()` 是**同步**的（LangGraph 的同步接口），
里面还会做 CPU 密集的事（BM25、numpy 点积）和**阻塞的 HTTP**（调模型）。

直接在 async 函数里调它，会**把整个事件循环堵死**：

- 一个用户在算检索，**别的所有用户**的请求（包括"我刚发的那条卡住了吗"这类心跳）
  全部停住；
- 更糟的是**流式输出根本没用了**：字是在工作线程里吐出来的，
  而事件循环正忙着等这个同步调用返回，于是所有字会在最后一起涌出去。
  **看起来像流式坏了，其实是事件循环被占了。**

所以用 `asyncio.to_thread` 把同步的图丢给线程池，
再用 `loop.call_soon_threadsafe` 把事件从工作线程送回事件循环。
这是"同步库 + 异步框架"的标准接法，**不是权宜之计**。

## 一个必须说清楚的取舍：客户端断开之后

用户关了页面 / 网络断了，这个生成器会被关掉，我们能做的是**不再往队列里塞事件**。
但**已经发出的那次模型调用不会停**——Python 的线程没法从外面杀掉，
而且真的杀了也会把连接池搞坏。所以这一次调用**仍然会跑完、仍然会计费**。

要做真取消，得让 LLM 客户端支持"取消令牌"并主动关掉 HTTP 流，
那是 `llm.py` 的活，不是这一层的。**这里如实说明，而不是假装没这回事**——
面试被问"用户断线你怎么处理"时，答"我复用了 asyncio 的取消"却其实
留着后台线程烧钱，比直接说"我没做取消"扣分更多（前者是错的，后者只是没做）。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from juris_agent import PROJECT_ROOT
from juris_agent.api.events import (
    Event,
    as_dict,
    events_from_update,
    slots_from_provenance,
)
from juris_agent.api.runtime import Runtime, build_runtime
from juris_agent.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CitationOut,
    CostOut,
    MemoryOut,
    SlotProvenanceOut,
    TraceOut,
)
from juris_agent.api.sessions import Session, SessionConflict
from juris_agent.observability import (
    bind,
    configure_logging,
    estimate_cost,
    get_logger,
    new_request_id,
)

log = get_logger(__name__)

# 推理模型的思考过程可能上千字。**逐字推它是纯粹的浪费**——
# 前端只显示一行"思考中…"，所以只要知道"还在想"就够了。
# 每攒够这么多字推一次，既能让前端动起来，又不会把队列灌满。
_THINKING_CHUNK_CHARS = 120


# --------------------------------------------------------------------------- #
# 单轮问答：async 和 SSE 两条路共用这一份逻辑
# --------------------------------------------------------------------------- #


async def _run_turn(
    rt: Runtime,
    session: Session,
    question: str,
    *,
    history: list[dict[str, str]] | None,
    stream_llm: bool,
    cancel: list[bool] | None = None,
    request_id: str = "",
) -> AsyncIterator[Event]:
    """跑一轮，边跑边吐事件。**非流式接口把事件丢掉、只留最后一个。**

    这样组织的好处是"两条路走的是同一份代码"：会话加锁、版本号校验、
    成本记账、错误处理只有一处实现。**分别写两份的话，
    迟早会出现"流式那条路忘了校验版本号"这种只在某一条路上存在的 bug。**

    ## 为什么 `request_id` 要能由调用方传进来

    ⚠️ **这是一个真 bug 的修法。** 原来的写法是这里自己 `new_request_id()`，
    而调用方为了给日志打上下文，又**自己 `new_request_id()` 了一次**。
    两个 id 都正常生成、都长得一样，但**是两个不同的字符串**。

    后果很隐蔽：用户界面底部显示的"请求 xxx"（来自 `done` 事件，是**这里**这个 id）
    和服务端日志里的 `request_id`（是**外面**那个）**对不上**。
    用户把界面上那串字符发给你排查，grep 日志**一条都搜不到**——
    而日志里确实有一批同格式的 id 在跑，所以看起来"日志是好的，
    是这个人抄错了"。**这类问题最耗时间的地方在于它把矛头指向了错误的嫌疑人。**

    现在由调用方生成一次、传进来：调用方能拿它 `bind()` 到日志上下文，
    这里拿它塞进 `done` 事件，**同一个串**。
    """
    trace_id = request_id or new_request_id()

    # --- 会话层面的准备，全部在事件循环里做完（不进工作线程）----------- #
    expected_version = session.version
    hist = history if history is not None else session.messages()
    turn = session.turn + 1

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Event | None] = asyncio.Queue()

    def emit(ev: Event) -> None:
        """从**工作线程**里把事件送回事件循环。

        `call_soon_threadsafe` 是唯一正确的过法：直接 `queue.put_nowait`
        从别的线程调，asyncio 的队列内部状态会被并发修改——
        症状是"偶发丢事件"，而且只在压力大时出现。
        """
        if cancel and cancel[0]:
            return
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    think_chars = 0

    def on_delta(piece: str) -> None:
        emit(Event("answer", {"text": piece}))

    def on_reasoning(piece: str) -> None:
        nonlocal think_chars
        think_chars += len(piece)
        if think_chars >= _THINKING_CHUNK_CHARS:
            think_chars = 0
            emit(Event("thinking", {}))

    final: dict[str, Any] = {}

    def work() -> None:
        """工作线程里跑的那一段。**同步代码，不要在里面对事件循环做任何假设。**"""
        try:
            llm = rt.llm
            # 开账本。**必须在工作线程里调**——它记的是"当前线程"的账，
            # 在事件循环线程里调，记的就是另一个线程，永远读到 0。
            if llm is not None and hasattr(llm, "begin_turn"):
                llm.begin_turn()

            state = None
            for mode, payload in rt.runner.stream(
                question,
                history=hist,
                memory=session.memory,
                on_delta=on_delta if stream_llm else None,
                on_reasoning=on_reasoning if stream_llm else None,
                # 要一路"全量状态"：最终答案、引用、tokens 都只在
                # 累积之后的状态里。理由见 `graph.AgentRunner.stream`。
                with_values=True,
            ):
                if mode == "updates":
                    for node, delta in (payload or {}).items():
                        for ev in events_from_update(node, delta or {}):
                            emit(ev)
                elif mode == "values":
                    state = payload
            final["state"] = state
            # ⚠️ **账必须在这里抄下来，不能出去再读。**
            #
            # `llm.turn_usage` 读的是**线程本地**的账本：哪个线程调的
            # `begin_turn()`，只有那个线程读得到。而下面 `_cost_of` 是在
            # **事件循环线程**里跑的（`work()` 已经返回了），在那边读
            # 只会拿到一本空账——**而且它不报错**，因为 `turn_usage`
            # 对"没开过账"的线程设计成返回全 0（脚本、单测里没人会去开账）。
            #
            # 后果是成本报表里 `prompt_tokens` 永远是 0、金额走"按总 token 粗估"
            # 那条退路，并附一句"未取到分项"。**那个数看着是合理的**——
            # 所以这个 bug 在真机上跑出来的第一版里就存在，却没被发现。
            # 直到我盯着 `basis` 那一行觉得"为什么会走退路"才挖出来。
            if llm is not None:
                final["usage"] = getattr(llm, "turn_usage", None)
        except Exception as e:  # noqa: BLE001
            # 兜住一切。**这个 except 不是在掩盖错误**——错误已经被
            # 原样打包成 `error` 事件发给前端了，而且下面还会记一条日志。
            # 不做这一层的话，异常会让 SSE 连接**没有任何说明地断掉**，
            # 前端只能显示"网络错误"，而真正的原因（比如索引没建）
            # 只留在服务端日志里，用户和演示者都看不到。
            log.exception("这一轮失败", session_id=session.session_id, request_id=trace_id)
            emit(Event("error", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            emit(None)  # 哨兵：告诉消费端"没了"

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield ev
    finally:
        # 无论是正常结束还是前端断连，都要等线程收尾。
        # **不等的话**，线程还在写 `final`，而我们已经去读它了。
        with contextlib.suppress(Exception):
            await task

    state = final.get("state")
    if state is None:
        # 走到这里说明线程出了没被兜住的错（理论上不会）。
        yield Event("error", {"message": "这一轮没有产出任何状态"})
        return

    cost = _cost_of(rt, state, final.get("usage"))
    # 写回会话。**在这里（事件循环线程里）做，不在工作线程里做**——
    # 会话表是个 dict，两个线程同时改它就是经典的静默数据竞争。
    answer = _g(state, "answer", "")
    try:
        rt.sessions.commit(
            session, expected_version=expected_version, question=question, answer=answer
        )
        warnings: list[str] = []
    except SessionConflict as e:
        # **不重试、不吞掉**：这一轮的回答照常给用户（钱都花了），
        # 但**不写回会话**——写回会覆盖掉别人的改动。
        # 代价是用户下一轮会少看到这一轮的历史，所以必须明确告诉他。
        log.warning("会话写回被拒", session_id=session.session_id, request_id=trace_id)
        warnings = [str(e)]

    # ⚠️ **一轮结束必须留一行日志。** 这一行是补上去的，理由很直接：
    #
    # 没有它的时候，"拿 request_id 去 grep 日志"这件事在**顺利的那一轮**
    # 是**空转的**——因为整条正常路径一条日志都不打（只在出错、会话写回冲突、
    # 客户端断开这三处才打）。于是 `request_id` 这个字段在实际使用中的
    # 唯一效果是"界面上多了一串字符"：用户说"我这轮回答很怪"，
    # 你拿着那串 id 去搜，**什么都搜不到**，日志里也没有任何东西能证明
    # 那一轮真的发生过。**一个只在出事时才有内容的排查机制，
    # 在最需要它的"说不出哪里不对"的场景里是哑的。**
    #
    # ⚠️ **这一行故意不传 `request_id` / `session_id`**：它们由 `bind()`
    # 从上下文自动注入（见 `observability/context.py`）。
    # 这不是省事，是**让 `bind()` 变成会出错的东西**——
    # 哪天有人忘了在外层 `bind()`，这一行就会少两个字段，
    # 而不是安静地打出一条永远正确的日志。
    log.info(
        "一轮结束",
        turn=turn,
        route=_g(state, "route", ""),
        finish=_g(state, "finish_reason", ""),
        tokens=cost.prompt_tokens + cost.completion_tokens if cost else 0,
        cny=round(cost.total_cny, 6) if cost else 0.0,
        billable=bool(cost.billable) if cost else False,
        warnings=len(warnings),
    )

    yield Event(
        "done",
        _to_response(
            state, session, turn=turn, cost=cost, warnings=warnings, request_id=trace_id
        ).model_dump(),
    )


def _cost_out(cost: Any) -> CostOut | None:
    """`observability.CostBreakdown` → 对外的 `CostOut`。

    ## 为什么要单独一个函数，而不是原地一行

    因为这里有一个**会静默出错**的转换：`CostOut` 是手工抄的镜像，
    pydantic 默认**忽略没声明的字段**。所以 `CostBreakdown` 上加了字段
    而 `CostOut` 忘了加，结果是"接口里那个字段凭空消失"——
    不报错、不告警、没有测试会红。写成函数至少**只有一个地方要改**，
    而且下面这个 `total_tokens` 的坑就是在这里被抓出来的。

    ## `total_tokens` 为什么必须显式传

    `CostBreakdown` 里**没有** `total_tokens` 这个字段（它只存"入/出/思考"
    三个分项——详见 `observability/cost.py` 里关于"思维链不能重复计"的那段）。
    而 `CostOut` 里声明了它。
    不显式传的话，`model_dump()` 里没有这个键，pydantic 就拿默认值 `0`，
    于是**接口每次都返回 `total_tokens: 0`**——这个字段在契约里躺着，
    看着像真的，实际上是死的。前端没用它，所以一直没暴露。
    """
    if cost is None:
        return None
    dumped = cost.model_dump()
    return CostOut(
        **dumped,
        # 思维链**已经算在 completion_tokens 里**了，不能加第三遍。
        total_tokens=(
            int(dumped.get("prompt_tokens", 0)) + int(dumped.get("completion_tokens", 0))
        ),
    )


def _cost_of(rt: Runtime, state: Any, usage: Any = None) -> Any:
    """这一轮的 token 与金额。

    ⚠️ `usage` 是**工作线程里抄下来的**那一本账（见 `work()` 里的注释），
    **不是在这里现读 `llm.turn_usage`**。在这里读的话拿到的是一本空账：
    账本是线程本地的，而这里跑在事件循环线程里。

    ⚠️ 也不用 `usage_total` 的前后差值——那在并发下会把别人的花销算进来。
    详见 `llm.LLMClient.begin_turn` 的 docstring。
    """
    model = rt.settings.model_gen

    if usage is not None and (usage.total_tokens or usage.prompt_tokens):
        return estimate_cost(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_hit_tokens=usage.prompt_cache_hit_tokens,
            cache_miss_tokens=usage.prompt_cache_miss_tokens,
        )

    # 退路：没有 LLM（拒答路径、闲聊路径、或者装配失败）时，
    # 用状态里记的总 token 估一个。**它只是个粗估**，所以金额那里
    # 会如实写上"按状态里的 token 总数估算"——报表上不能出现
    # 一个和别行算法不同的数却不说明。
    #
    # ⚠️ **走到退路就说明"这一轮没有计过费的调用"**，所以要标 `billable=False`。
    # 这条退路最常见的触发者不是拒答，而是 `replay` 模式：
    # `LLMClient` 命中缓存时**刻意不记账**（见 `llm.py` 里那段
    # "不要累加 hit.usage"），于是 `turn_usage` 是一本空账，代码就掉到这里。
    # 但状态里的 `tokens_used` 还留着**当初真调时**的数（17255），
    # 于是金额算出 ¥0.138——**一个接口都没打，却报了一毛三**。
    # 这个数看着太正常了，肉眼绝对看不出来；所以不靠调用方自觉，
    # 而是把"没计费"做成一个显式字段。
    total = int(_g(state, "tokens_used", 0) or 0)
    c = estimate_cost(model=model, prompt_tokens=0, completion_tokens=total)
    c.billable = False
    c.basis = (
        "按状态里记录的 token 总数估算（未取到分项）"
        if total
        else "这一轮没有调用模型"
    )
    return c


def _to_response(
    state: Any,
    session: Session,
    *,
    turn: int,
    cost: Any = None,
    warnings: list[str] | None = None,
    request_id: str = "",
) -> ChatResponse:
    """内部状态 → 对外响应。**只有这一个地方做这个转换**（见 `schemas.py`）。

    ⚠️ 全部字段都走 `_g()` 读，不用 `getattr`——理由见 `_g` 的 docstring。
    这不是风格偏好：**用 `getattr` 读一个 dict 不报错，它返回默认值**，
    于是"答案"会安安静静地变成空串，接口返回 `200 OK` 加一个空回答。
    """
    injected = {a.index: a for a in (_g(state, "injected", []) or [])}
    report = _g(state, "citation_report", None)

    if report is not None:
        r = as_dict(report)
        cited = [injected[i] for i in r.get("valid", []) if i in injected]
    else:
        # 没有校验报告（拒答/闲聊路径）时退回按 key 匹配。
        keys = set(_g(state, "citations", []) or [])
        cited = [a for a in injected.values() if a.article_key in keys]

    memory = None
    if _g(state, "memory_policy", "") or _g(state, "memory_block", ""):
        memory = MemoryOut(
            policy=_g(state, "memory_policy", ""),
            tokens=int(_g(state, "memory_tokens", 0) or 0),
            layers=list(_g(state, "memory_layers", []) or []),
            slots=[
                SlotProvenanceOut(**s)
                for s in slots_from_provenance(_g(state, "slots_provenance", {}) or {})
            ],
            dropped=list(_g(state, "memory_dropped", []) or []),
            missing=list(_g(state, "missing_slots", []) or []),
        )

    return ChatResponse(
        session_id=session.session_id,
        turn=turn,
        answer=_g(state, "answer", ""),
        citations=[_citation(a) for a in cited],
        route=_g(state, "route", ""),
        route_reason=_g(state, "route_reason", ""),
        refusal=_g(state, "refusal", None),
        finish_reason=_g(state, "finish_reason", ""),
        finish_detail=_g(state, "finish_detail", ""),
        citation_report=as_dict(report) if report is not None else None,
        memory=memory,
        cost=_cost_out(cost),
        trace=[_trace(t) for t in (_g(state, "trace", []) or [])],
        guard_hits=[as_dict(g) for g in (_g(state, "guard_hits", []) or [])],
        errors=[str(e) for e in (_g(state, "errors", []) or [])],
        warnings=list(warnings or []),
        request_id=request_id,
    )


def _g(state: Any, key: str, default: Any) -> Any:
    """从状态里读一个字段，**dict 和对象两种形状都吃**。

    ## 为什么必须这样读（这是实测出来的，不是"防御性编程"）

    `AgentRunner.stream(with_values=True)` 拿到的全量状态，
    **实测是一个普通 `dict`，不是 `AgentState` 实例**：

    ```python
    for mode, payload in run.stream("...", with_values=True):
        if mode == "values":
            print(type(payload))     # <class 'dict'>   ← 不是 AgentState
    ```

    而且它**会漏键**：一次实测里 `AgentState` 声明了 35 个字段，
    `values` 给出的 dict 里少了 `refusal`（那一轮的 `finish_reason` 是 `refused`，
    说明这个键本该有值）。这意味着**`payload["refusal"]` 会抛 `KeyError`，
    而 `getattr(payload, "refusal", None)` 不抛——它返回 None**。

    最坏的一种组合是 `getattr(payload, "answer", "")`：
    它不报错、返回空串，接口照样返回 200，
    **前端拿到的是一条永远空着的回答**。这种 bug 在只看日志的时候
    完全看不出来（没有任何异常），只能靠"盯着屏幕发现答案是空的"。

    所以这里统一走一个容忍两种形状、且对缺键返回默认值的读法。
    它比 `getattr` 多花几纳秒，换掉的是"静默返回空答案"这个后果。

    ⚠️ 另外**不能用 `if key in state` 之类的方式区分"没这个键"和"值是 None"**——
    本项目里这两者的处理是一样的（都当默认值）。区分它们只会让
    调用方为每个字段多写一个分支，而没有一个字段真的需要区分。
    """
    if isinstance(state, dict):
        v = state.get(key, default)
    else:
        v = getattr(state, key, default)
    return default if v is None else v


def _fields(obj: Any, model: type[Any]) -> dict[str, Any]:
    """把 pydantic 对象或 dict 的字段挑出来，只留目标模型认识的。

    `values` 里的嵌套元素**仍然是 pydantic 对象**（`InjectedArticle`、
    `TraceStep`、`GuardHit`……），只有最外层变成了 dict。
    **"外层是 dict、里层是对象"这个混合形状是实测出来的**，
    写代码时很容易假设"全变 dict 了"或"全没变"，两种假设都会在某些路径上错。

    只挑目标模型认识的字段，是为了**允许状态比响应模型多字段**：
    以后给 `TraceStep` 加一个内部字段时，这里不用改也不会报错。
    """
    d = as_dict(obj)
    return {k: v for k, v in d.items() if k in model.model_fields}


def _citation(a: Any) -> CitationOut:
    return CitationOut(**_fields(a, CitationOut))


def _trace(t: Any) -> TraceOut:
    return TraceOut(**_fields(t, TraceOut))


# --------------------------------------------------------------------------- #
# 应用
# --------------------------------------------------------------------------- #


def create_app(runtime: Runtime | None = None, *, serve_web: bool = True) -> FastAPI:
    """造应用。**`runtime` 可注入，测试靠它绕开模型和 API key。**

    ## ⚠️ 为什么 `runtime=None` 时**不在这里**装配

    最省事的写法是开头就写 `rt = runtime or build_runtime()`。**不做**，因为：

    `build_runtime()` 会加载检索器——也就是那个 1GB 的嵌入模型，
    通常还要建一次索引，**几秒到几分钟**。而模块底部有
    `app = create_app()`，于是**任何 `import juris_agent.api.app` 都会付这个代价**：

    - `pytest` 里只 import 一下就要先等模型加载，测试从 11 秒变成几分钟；
    - `uvicorn --reload` 每改一行代码都要重新加载模型，改一行等半分钟；
    - 只想 `import` 一下看接口定义（读文档、查字段）也得先等。

    **这个代价买不到任何东西**：真正需要依赖的时刻是"服务器开始收请求"，
    也就是 lifespan 启动的那一刻。

    所以装配挪进 lifespan，`app.state.runtime` 启动前是 `None`。
    `import` 只造了一堆路由，**不碰模型、不读 API key、不建索引**。
    测试注入 `runtime=` 时连 lifespan 都不用走。
    """
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 注入过就用注入的；没注入才真的装配。**这就是"测试不加载模型"的全部秘密。**
        if app.state.runtime is None:
            app.state.runtime = build_runtime()
        rt: Runtime = app.state.runtime
        configure_logging(level="INFO", json=rt.settings.is_prod)
        log.info("服务启动", ready=rt.ready, problems=rt.problems)
        try:
            yield
        finally:
            rt.close()
            log.info("服务停止")

    app = FastAPI(
        title="juris-agent",
        version="0.1.0",
        description=(
            "可溯源劳动法咨询 Agent。答案一律附法条编号与原文，"
            "超出范围时明确拒答。**输出不构成法律意见。**"
        ),
        lifespan=lifespan,
    )
    app.state.runtime = runtime  # 可能是 None，lifespan 里补上

    def rt_of(request: Request) -> Runtime:
        """从**请求对象**上取 runtime，不闭包捕获 `rt`。

        差别在哪：闭包捕获的话，测完第一个用例想把假 runtime 换掉，
        得重新造一个 app；从 `app.state` 读的话，测试改一行
        `app.state.runtime = other` 就换掉了。**也让"注入"和"lifespan 里装配"
        走的是同一条读取路径**——两条路径写两份的话，
        迟早出现"注入时能跑、真启动时不行"。
        """
        rt: Runtime | None = getattr(request.app.state, "runtime", None)
        if rt is None:
            # 只有一种情况会走到这里：没走 lifespan 就直接调接口
            # （比如用 `httpx.ASGITransport` 但忘了触发 startup）。
            raise HTTPException(
                status_code=503,
                detail={"message": "服务未初始化（lifespan 没跑）", "problems": []},
            )
        return rt

    # ---- 健康检查 ------------------------------------------------------ #

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        """**先看这个。** 前端报错时，第一步就是确认服务是不是真的答得了问题。

        ⚠️ 它**不挂 `response_model`**，理由见 `schemas.HealthOut` 的 docstring：
        严格校验会让以后新增的 health 字段被静默丢掉。
        """
        return rt_of(request).health()

    # ---- 非流式问答 ---------------------------------------------------- #

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        """一问一答，等全部算完再返回。**给脚本、curl、压测用。**

        ⚠️ 浏览器里的产品界面**不该**用这个——它会盯着转圈等七八秒。
        用 `/api/chat/stream`。
        """
        rt = rt_of(request)
        if rt.runner is None:
            raise HTTPException(status_code=503, detail={"problems": rt.problems})

        # ⚠️ **同一个 id 两处用**：`bind()` 给它打日志上下文，
        # `_run_turn` 把它放进 `done` 事件。分两次生成的话，
        # 界面上的请求 id 在日志里搜不到（详见 `_run_turn` 的 docstring）。
        #
        # ⚠️ **`bind` 分两层套**，不是写啰嗦了：外层只绑 `request_id`，
        # 因为**建会话这一步也要落日志**（"新建会话"），而 `session_id`
        # 得等建完才知道。不这么套的话，"新建会话"那一条是**唯一一条
        # grep 不到 request_id 的日志**——于是同一个请求的日志被劈成两半：
        # 一半用 id 能搜到，一半得靠时间戳猜。`bind` 支持只传一个、
        # 也支持嵌套（见它的 docstring），这里正是它说的那种用法。
        rid = new_request_id()
        with bind(request_id=rid):
            session = rt.sessions.get_or_create(req.session_id)
            hist = [m.model_dump() for m in req.history] if req.history is not None else None
            with bind(session_id=session.session_id):
                async with session.lock:
                    done: Event | None = None
                    async for ev in _run_turn(
                        rt,
                        session,
                        req.question,
                        history=hist,
                        stream_llm=False,
                        request_id=rid,
                    ):
                        if ev.name == "done":
                            done = ev
                        elif ev.name == "error":
                            # 非流式路径上错误要变成 HTTP 状态，而不是混在正常响应里。
                            # **混进去的话，调用方得先解析 JSON 才知道成功了没有。**
                            raise HTTPException(status_code=500, detail=ev.data)
                    if done is None:
                        raise HTTPException(status_code=500, detail={"message": "没有产出结果"})
                    return ChatResponse.model_validate(done.data)

    # ---- 流式问答（SSE）------------------------------------------------- #

    async def _sse(rt: Runtime, req: ChatRequest, request: Request) -> AsyncIterator[str]:
        """⚠️ `rt` 是**参数**，不是在生成器里现取的。

        原因：这是个 **async generator**——它的函数体要等到开始迭代才执行。
        在函数体里抛 `HTTPException` 不会变成 503 响应，
        而是会变成"响应头已经发出去了、流中途炸掉"。
        所以取 runtime 这一步放在外面那个普通 `async def` 里做，
        那里抛异常才是正常的 HTTP 错误。**这也顺带把"取 rt"这件事
        从三条路径收敛成同一个入口。**
        """
        if rt.runner is None:
            # 装配失败也要用**事件**说，不能直接断开——
            # 不然前端只能显示"网络错误"，看不到真正缺什么。
            yield Event("error", {"message": "服务未就绪", "problems": rt.problems}).to_sse()
            return

        cancel = [False]

        # 和 `_run_turn` 里用的是**同一个 id**，理由见那里。
        #
        # ⚠️ **两层 `bind`，和 `/api/chat` 里一样**：外层先绑 `request_id`
        # （"新建会话"那条日志也要能 grep 到 id），建完会话再补 `session_id`。
        # 这两个 `with` 会**跨着 `yield` 一直挂着**——这是故意的，
        # 也是 async generator 里用 `bind` 的正常姿势：流没结束之前，
        # 这个请求的日志上下文就该一直有效。生成器被提前关掉
        # （客户端断开、`StreamingResponse` 被取消）时，`with` 会随栈展开
        # 自动还原，不需要我们手动善后。
        rid = new_request_id()
        with bind(request_id=rid):
            session = rt.sessions.get_or_create(req.session_id)
            hist = [m.model_dump() for m in req.history] if req.history is not None else None
            with bind(session_id=session.session_id):
                async with session.lock:
                    try:
                        async for ev in _run_turn(
                            rt,
                            session,
                            req.question,
                            history=hist,
                            stream_llm=req.stream_llm,
                            cancel=cancel,
                            request_id=rid,
                        ):
                            yield ev.to_sse()
                            if await request.is_disconnected():
                                cancel[0] = True
                                log.info("客户端断开，停止推送", session_id=session.session_id)
                                return
                    except asyncio.CancelledError:
                        cancel[0] = True
                        raise

    @app.post("/api/chat/stream")
    async def chat_stream_post(req: ChatRequest, request: Request) -> StreamingResponse:
        """流式问答。**前端用这个**（问题是 POST body，不受 URL 长度限制）。"""
        rt = rt_of(request)  # 在这里取，抛异常才是正常的 HTTP 错误（见 `_sse`）
        return StreamingResponse(
            _sse(rt, req, request),
            media_type="text/event-stream",
            headers={
                # 这两行是给**反向代理**看的。少了它们，nginx 会把整个响应
                # 缓冲起来、等生成完才一次吐给浏览器——**流式就白做了**，
                # 而且症状是"本地好好的，一上线就不流了"，很难往代理上想。
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/chat/stream")
    async def chat_stream_get(
        request: Request,
        question: str = Query(min_length=1, max_length=2000),
        session_id: str | None = None,
        stream_llm: bool = True,
    ) -> StreamingResponse:
        """同一个流式接口的 GET 版本，**纯粹为了 `curl` 和 `EventSource` 方便**。

        验收标准里那条"`curl` 能看到事件按序流出"靠的就是它：
        POST 的流式接口用 curl 敲起来要带一堆参数和引号，
        而演示现场最不该出现的就是"命令敲错了"。

        ⚠️ 它和 POST 版本**共用同一个生成器**，所以行为完全一致。
        """
        rt = rt_of(request)
        req = ChatRequest(question=question, session_id=session_id, stream_llm=stream_llm)
        return StreamingResponse(
            _sse(rt, req, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- 法条原文 ------------------------------------------------------ #

    @app.get("/api/article/{article_key}")
    async def article(article_key: str, request: Request) -> dict[str, Any]:
        """按 `article_key`（`lcl#47`）取法条原文。

        ⚠️ 流式事件里**已经带了正文**（见 `events._from_retrieve`），
        所以前端的"点开看原文"不需要调这个接口。它存在是为了两件事：

        1. **深链**：把 `?article=lcl#47` 这样的链接发给别人，能直接打开原文；
        2. **核对**：评测/排查时想确认"编号 ↔ 原文"对不对得上。
        """
        rt = rt_of(request)
        by_key = getattr(rt.retriever, "articles_by_key", None)
        if not by_key:
            raise HTTPException(status_code=503, detail={"problems": rt.problems})
        art = by_key.get(article_key)
        if art is None:
            raise HTTPException(status_code=404, detail={"message": f"没有这条法条：{article_key}"})
        return {
            "article_key": art.article_key,
            # ⚠️ `law_name` 是法规全称（`中华人民共和国劳动合同法`），
            # `citation()` 用的是 `law_short`（`劳动合同法`）。
            # 两个都返回：全称给人看，简称是**引用时需要和答案里的编号一致**的那个。
            "law_name": art.law_name,
            "law_short": art.law_short,
            "article_no": art.article_no,
            "citation": art.citation(),
            "text": art.text,
            "chapter": art.chapter,
            "section": art.section,
            "effective_date": art.effective_date.isoformat() if art.effective_date else None,
            "version_date": art.version_date.isoformat() if art.version_date else None,
            "effective_date_source": art.effective_date_source,
        }

    # ---- 静态页面 ------------------------------------------------------ #

    if serve_web:

        @app.get("/")
        async def index(request: Request) -> Any:
            """产品界面。**单文件、无构建步骤**（理由见 `web/index.html` 顶部）。"""
            page = PROJECT_ROOT / "web" / "index.html"
            if not page.exists():
                rt: Runtime | None = getattr(request.app.state, "runtime", None)
                return JSONResponse(
                    status_code=404,
                    content={
                        "message": "web/index.html 不存在",
                        "problems": rt.problems if rt else [],
                    },
                )
            return FileResponse(page)

        @app.get("/favicon.ico")
        async def favicon() -> Any:
            # 不返回 404 是为了**让浏览器控制台干净**。
            # 演示时打开控制台是一片红，看起来像是项目有问题——
            # 而它其实只是少了个图标。
            return JSONResponse(status_code=204, content=None)

    return app


# uvicorn 的入口：`uvicorn juris_agent.api.app:app`
#
# ⚠️ 这一行**只是造路由，不装配依赖**（不加载模型、不读 API key、不建索引）。
# 真正的装配在 lifespan 里、服务真启动时才发生——理由见 `create_app` 的 docstring。
# 所以 `import juris_agent.api.app` 是**廉价且无副作用**的，
# 测试可以放心 import 完再注入一个假 runtime。
app = create_app()


__all__ = ["app", "create_app"]
