"""把节点连成图（F16）。

## 图的形状

```
                    ┌─────────────┐
        START ─────▶│   entry     │  路由（规则，不用模型）
                    └──────┬──────┘
             ┌─────────────┼──────────────┬──────────────┐
        law  │        calc │   out_of_scope│      chitchat│
             ▼             ▼              ▼              ▼
      ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
      │  retrieve  │  │ retrieve │  │  refuse  │  │ chitchat │
      └─────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
            │              ▼             │             │
            │        ┌──────────┐         │             │
            │        │   tool   │◀─┐      │             │
            │        └────┬─────┘  │      │             │
            │             │        │(还要调别的工具)     │
            └─────────────┴────────┴──────┴─────────────┘
                          │
                          ▼
                    ┌──────────┐   ← 四道闸门都在这一个节点里
                    │  guard   │
                    └────┬─────┘
              ┌──────────┴──────────┐
        还有下一步                 撞闸门/该收尾
              ▼                     ▼
        （回到对应节点）      ┌─────────────┐
                             │  finalize   │ ──▶ END
                             └─────────────┘
```

## 为什么 `guard` 只有一个、而且所有节点都经过它

四道防死循环的闸门（步数 / 重复调用 / 无进展 / token 预算）必须**无死角**。
如果每个工作节点后面各挂一个条件边自己判断，就有两个风险：

1. **漏一个**。以后加了新节点，忘了挂闸门，那个节点就是无限循环的入口。
2. **判据漂移**。四条闸门的逻辑复制成四份之后，改了一份忘了另外三份，
   于是"某些路径上步数上限是 12、另一些路径上是 20"——这种 bug 极难发现。

做成"**唯一出口 + 唯一闸门**"，图的结构本身保证了不可能漏。
代价是状态里多一个 `next_step` 字段（由工作节点写、闸门读）。

## 为什么用条件边而不是在节点里循环

`tool` 节点一次只走一轮工具调用，循环由**图**来转（`tool → guard → tool`）。
理由：循环控制权交给图之后，"最多转几圈"由一道闸门统一决定；
写在节点里的 `while` 必须自己实现一遍计数和退出条件，
而且**它不会出现在 trace 里**——出问题时你看到的是一步，实际跑了几十次。

## ⚠️ 关于 checkpointer（可恢复的分支）

`compile(checkpointer=...)` 是 LangGraph 的"存档"功能：每一步的状态存一份，
可以从中间某步恢复（`get_state` / `update_state`）。

本项目**把它做成可选项而不是默认开启**，理由：
- 开着的代价是**每次请求都要序列化整个状态并落盘**（本项目状态里带着完整法条正文，
  一次请求几十 KB），
- 而它的收益（人工干预、分支重放）在**单轮问答**里用不上。

真正需要它是**多轮会话**（F22）——那时会由 `api/` 层按 `thread_id` 传入。
所以这里留一个参数，调用方决定。
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from juris_agent.agent.nodes import AgentDeps, Nodes
from juris_agent.agent.state import AgentState

# 条件边可能去的地方。**列全了才能被 `path_map` 检查**——
# LangGraph 在编译时校验条件边的返回值是否都在 `path_map` 里，
# 少写一个会在运行时才炸（而且是在特定输入下才炸）。
_AFTER_ENTRY: dict[str, str] = {
    "law": "retrieve",
    "calc": "retrieve",
    "out_of_scope": "refuse",
    "chitchat": "chitchat",
}

# `guard` 之后的去向。**注意 `tool` 也在里面**——那是工具循环回到自己的边，
# 整个"最多跑几轮工具"完全由闸门（`max_steps` + `no_progress`）控制。
_AFTER_GUARD: dict[str, str] = {
    "retrieve": "retrieve",
    "tool": "tool",
    "generate": "generate",
    "refuse": "refuse",
    "chitchat": "chitchat",
    "finalize": "finalize",
}


def _route_after_entry(state: AgentState) -> str:
    """入口之后的四岔路。**路由节点已经把决定写进 `state.route` 了**，
    这里只做翻译，不做判断——判断逻辑集中在 `router.decide()` 一处。"""
    return state.route or "law"


def _route_after_guard(state: AgentState) -> str:
    """闸门之后的去向。

    `guard` 节点会把 `next_step` 设成 `"finalize"`（撞闸门时）或保持不变。
    **兜底成 `finalize`** 而不是 `"generate"`：如果某天有节点忘了写 `next_step`，
    停下来总比继续烧钱好——停下来的代价是一次不完整的回答，
    继续跑下去的代价可能是几百次调用。
    """
    return state.next_step or "finalize"


def build_graph(deps: AgentDeps, *, checkpointer: Any | None = None) -> Any:
    """装配并编译图。

    `deps` 里的东西（检索器、LLM、工具执行器）在**图之外**构造、注入进来。
    理由：它们都是有状态的重量级对象（模型、连接池、熔断器），
    每个请求重建一份会让熔断器和幂等缓存**永远空着**——那它们就白做了。
    **一次请求一个图、一份 deps 共享**是这里的分工。
    """
    nodes = Nodes(deps)
    g = StateGraph(AgentState)

    g.add_node("entry", nodes.entry)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("tool", nodes.tool)
    g.add_node("generate", nodes.generate)
    g.add_node("refuse", nodes.refuse)
    g.add_node("chitchat", nodes.chitchat)
    g.add_node("guard", nodes.guard)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "entry")
    g.add_conditional_edges("entry", _route_after_entry, _AFTER_ENTRY)

    # 三个"工作节点"的出口都指向同一个闸门。**这是本文件的核心结构。**
    for n in ("retrieve", "tool", "generate"):
        g.add_edge(n, "guard")
    g.add_conditional_edges("guard", _route_after_guard, _AFTER_GUARD)

    # 拒答和闲聊不经过闸门：它们不做循环，一步就结束，没有循环可防。
    # （它们仍然会被 `finalize` 的断言检查到。）
    g.add_edge("refuse", "finalize")
    g.add_edge("chitchat", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


class AgentRunner:
    """对外的调用口。**包住 `graph.invoke`，因为调用方不该关心图长什么样。**

    用法：

        runner = AgentRunner(AgentDeps(retriever=r, registry=reg, executor=ex, llm=llm))
        result = runner.ask("试用期最长多久？")
        print(result.answer, result.citations)

    `ask()` 只收一个字符串，返回填好的 `AgentState`。
    想要流式（SSE）就调 `runner.stream(...)`，它按步 yield。
    """

    def __init__(self, deps: AgentDeps, *, checkpointer: Any | None = None) -> None:
        self.deps = deps
        self.graph = build_graph(deps, checkpointer=checkpointer)

    def ask(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        thread_id: str | None = None,
    ) -> AgentState:
        """跑一轮。**返回完整状态**，包括 trace——
        调用方要展示"检索了哪些、引用了哪些"，这些都在里面。"""
        init = AgentState(question=question, history=list(history or []))
        cfg = {"configurable": {"thread_id": thread_id}} if thread_id else None
        out = self.graph.invoke(init, cfg) if cfg else self.graph.invoke(init)
        # LangGraph 在有 checkpointer 时返回 dict；无 checkpointer 时返回状态对象。
        # 两种都归一成 `AgentState`，免得调用方要判类型。
        return out if isinstance(out, AgentState) else AgentState.model_validate(out)

    def stream(self, question: str, *, history: list[dict[str, str]] | None = None):
        """按步产出 `(节点名, 该步的状态增量)`。

        ⚠️ **`stream_mode="updates"` 给的是"增量"不是"全量"**。
        SSE 层要的是"这一步发生了什么"（`tool_call`、`token`），
        增量正好对应；要全量得自己累积。这个 swap 在写 SSE 时很容易搞反，
        所以在这里写清楚，别到 `api/` 里再猜。
        """
        init = AgentState(question=question, history=list(history or []))
        yield from self.graph.stream(init, stream_mode="updates")


def make_runner(
    *,
    retriever: Any = None,
    registry: Any = None,
    executor: Any = None,
    llm: Any = None,
    settings: Any = None,
    logger: Any = None,
    checkpointer: Any | None = None,
) -> AgentRunner:
    """便利构造。

    **`registry` / `executor` 都不传时会自动造一份**（用内置工具、
    不注册 `search_law`），这样"只想试试图能不能跑"的场景不用先准备好语料。

    ⚠️ 自动造的那份 executor **是新的**，熔断器和幂等缓存也是新的。
    生产环境务必自己造一份**进程级共享**的传进来——
    否则熔断器永远学不到东西（每次都是全新的、没有历史失败的窗口）。
    """
    from juris_agent.config import get_settings

    s = settings or get_settings()
    if registry is None:
        from juris_agent.tools.builtin import build_registry

        registry = build_registry(retriever)
    if executor is None:
        from juris_agent.tools.executor import ToolExecutor

        executor = ToolExecutor(registry, s)
    deps = AgentDeps(
        settings=s,
        registry=registry,
        executor=executor,
        retriever=retriever,
        llm=llm,
        logger=logger,
    )
    return AgentRunner(deps, checkpointer=checkpointer)


__all__ = ["AgentRunner", "build_graph", "make_runner"]
