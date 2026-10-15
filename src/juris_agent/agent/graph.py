"""把节点连成图（F16）。

## 图的形状

```
                    ┌─────────────┐
        START ─────▶│  remember   │  记忆（槽位/窗口/摘要/长期召回）
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   entry     │  路由（规则，不用模型）
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

## `remember` 为什么在最前面、而且**不经过闸门**

**在最前面**：路由和工具循环都要看它产出的事实（第 20 轮的月薪、
用户的工作年限）。放在 `entry` 后面也行——但路由将来要看槽位
（"上下文里有解除情形就该走计算路"），到时候再挪顺序，
一定会漏掉某个分支。**顺序先定对，比将来挪一次安全。**

**不经过闸门**，原因很直接：**闸门的唯一职责是回答"下一步去哪"。**
`remember` 只有一个后继（`entry`），没有分支可判。
让它绕一圈闸门，换来的是"多一条恒真的边"——那不是防护，那是装饰。

⚠️ 但它**算一步**（`steps` 加 1）。口径统一比省下那一圈预算重要：
`steps` 的含义是"这条轨迹走了几步"，不是"调了几次模型"。
两种口径混在一个字段里，报表上"平均几步跑完"这个数就没法解释了。

再想往图上加节点时先问一句：**它有几个后继？**
只有一个的话，它不需要走闸门。

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

    g.add_node("remember", nodes.remember)
    g.add_node("entry", nodes.entry)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("tool", nodes.tool)
    g.add_node("generate", nodes.generate)
    g.add_node("refuse", nodes.refuse)
    g.add_node("chitchat", nodes.chitchat)
    g.add_node("guard", nodes.guard)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "remember")
    g.add_edge("remember", "entry")
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

    def new_memory(self) -> Any:
        """**开一段新会话时该调这个**：造一个干净的记忆管理器。

        单独留一个方法，是因为"会话"这个概念目前只体现在这里。
        将来接了 FastAPI，会话表里存的就是这个对象
        （按 `thread_id` 存，一个会话一个）。
        """
        from juris_agent.memory.manager import manager_from_settings

        return manager_from_settings(self.deps.settings)

    def _memory_for(self, memory: Any, history: list[dict[str, str]]) -> Any:
        """决定这一轮用哪个记忆管理器。

        ## 默认行为的取舍（值得说清楚，因为它是本项目唯一一个"隐式约定"）

        **不传 `memory` 时，每轮现造一个新的。** 这看着浪费，其实是对的：

        - 结果是**正确的**。`MemoryManager.catch_up()` 会从完整历史里
          补齐槽位，所以"每轮新建"和"跨轮复用"算出来的槽位表一样。
        - 代价是每轮把历史重抽一遍正则（20 轮规模下几十毫秒），
          **换掉的是一个很脏的失败模式**：进程里共用一个管理器，
          A 用户的月薪会出现在 B 用户的答案里。这种串数据一旦发生，
          日志里完全看不出来。

        **想省那几十毫秒就自己传**：一个会话一个实例，由调用方持有。
        这是显式的优化，出问题的范围也明确（"你把两个会话指向同一个了"）。
        """
        if memory is not None:
            return memory
        if self.deps.memory is not None:
            return self.deps.memory
        if not self.deps.settings.memory.enabled:
            return None
        return self.new_memory()

    def ask(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        memory: Any = None,
        thread_id: str | None = None,
        on_delta: Any = None,
        on_reasoning: Any = None,
    ) -> AgentState:
        """跑一轮。**返回完整状态**，包括 trace——
        调用方要展示"检索了哪些、引用了哪些"，这些都在里面。

        `memory` 给多轮会话用：把同一段会话的每一轮都指向同一个管理器，
        省掉每轮重抽历史的开销。**不同会话之间绝不能共用**——
        理由写在 `_memory_for` 里。

        `on_delta` / `on_reasoning` 是**流式输出**用的回调（F29）：
        传了它们，答案就是边生成边交出去的，而不是等生成完再一次给。
        **一轮只影响这一轮**——它们进 `AgentState`，不进 `deps`，
        所以并发请求之间不会串（理由见 `state.py` 上那两个字段）。
        """
        hist = list(history or [])
        init = AgentState(
            question=question,
            history=hist,
            memory=self._memory_for(memory, hist),
            on_delta=on_delta,
            on_reasoning=on_reasoning,
        )
        cfg = {"configurable": {"thread_id": thread_id}} if thread_id else None
        out = self.graph.invoke(init, cfg) if cfg else self.graph.invoke(init)
        # LangGraph 在有 checkpointer 时返回 dict；无 checkpointer 时返回状态对象。
        # 两种都归一成 `AgentState`，免得调用方要判类型。
        return out if isinstance(out, AgentState) else AgentState.model_validate(out)

    def stream(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        memory: Any = None,
        on_delta: Any = None,
        on_reasoning: Any = None,
        with_values: bool = False,
    ):
        """按步产出 `(节点名, 该步的状态增量)`。

        ⚠️ **`stream_mode="updates"` 给的是"增量"不是"全量"**。
        SSE 层要的是"这一步发生了什么"（`tool_call`、`token`），
        增量正好对应；要全量得自己累积。这个 swap 在写 SSE 时很容易搞反，
        所以在这里写清楚，别到 `api/` 里再猜。

        ⚠️ 流式模式下**调用方要自己把增量拼起来**才能拿到
        `memory_block` / `injected` 这些跨节点的字段。SSE 层真正要的是
        "这一步产出了什么事件"，不是"现在的完整状态"。

        ⚠️ **这个方法只是"按节点分步"，不等于"逐字"。** 真正的逐字流
        来自 `on_delta`（它在一次 `next()` 内部被调用很多次，都在同一个节点里）。
        两者是**两个不同粒度**，SSE 层要同时转发：见 `api/events.py`。

        ## `with_values=True`：连全量状态一起要

        `updates` 模式有个绕不开的限制：**它不给你最终状态**。
        请求结束时"答案是什么、引用了哪几条、撞没撞闸门"这些要报给用户的字段，
        全都在**累积之后**的状态里，而不在任何单个增量里。

        自己把增量拼起来是**错的**（而且错得很隐蔽）：状态里那些
        `Annotated[..., operator.add]` 的字段要按 `+` 合并，普通字段要覆盖、
        `None` 要跳过——这就是 LangGraph 的 reducer。**自己再实现一遍 reducer，
        等于把"合并规则"复制成两份**，改了一份忘了另一份，
        表现为"trace 里少了几步"这种没人会想到去查的偏差。

        所以要全量时用这个开关：多要一路 `values`，
        让 LangGraph 把它已经算好的全量状态直接给你。
        代价是**每步都要序列化一遍完整状态**（本项目状态里带着法条正文，
        一次几十 KB），所以**默认关着**——离线评测那种跑几千次的场景不该付这个钱。

        开了之后产出变成 `(mode, payload)` 二元组，`mode` 是 `"updates"` 或
        `"values"`。这个形状变化放在参数后面而不是默认行为里，
        是因为调用方**必须显式处理两种形状**，而默认形状保持简单。
        """
        hist = list(history or [])
        init = AgentState(
            question=question,
            history=hist,
            memory=self._memory_for(memory, hist),
            on_delta=on_delta,
            on_reasoning=on_reasoning,
        )
        if with_values:
            yield from self.graph.stream(init, stream_mode=["updates", "values"])
        else:
            yield from self.graph.stream(init, stream_mode="updates")


def make_runner(
    *,
    retriever: Any = None,
    registry: Any = None,
    executor: Any = None,
    llm: Any = None,
    settings: Any = None,
    logger: Any = None,
    memory: Any = None,
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
        memory=memory,
    )
    return AgentRunner(deps, checkpointer=checkpointer)


__all__ = ["AgentRunner", "build_graph", "make_runner"]
