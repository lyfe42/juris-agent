"""Agent 的「局面」——每一步读写的那份状态（F14）。

## 先说清楚：为什么要有"状态"这个东西

你在 LangChain 里写的 `chain = prompt | model | parser` 是**一条直线**：
数据从左边流到右边，流完就没了。这条直线没法表达三件事：

1. **回退**。工具报错了要改参数重试——直线只能往下走，回不去。
2. **分岔**。这是法条问题还是计算题？不同的问题要走不同的路。
3. **记账**。已经走了几步、烧了多少 token、调用过哪些工具——
   这些信息不属于任何一步，但每一步都要读、都要写。

状态对象就是那张**所有节点都能读写的白板**。节点（node）拿到白板、
做一件事、在白板上写几笔；边（edge）读白板决定下一步去哪。
LangGraph 干的就是"把白板传下去"这件事。

> 面试被问「为什么不用 LangChain 的 Chain」时，答案就是上面三段话。
> 别说"因为 LangGraph 更高级"——那是没想过。

## 白板上都有什么（按用途分四组）

- **输入**：`question`、`history`
- **中间产物**：`route`、`injected`、`tool_results`、`draft`
- **记账**：`steps`、`tokens_used`、`trace`、`progress_keys`（防死循环用，见 `guard.py`）
- **输出**：`answer`、`citations`、`citation_report`、`refusal`、`finish_reason`
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# 路由的四种去向。**这个字面量是图里条件边的判断依据**，
# 所以它必须只有这几种取值——多一个值就会出现"没有边接得住"的死胡同。
Route = Literal["law", "calc", "out_of_scope", "chitchat"]

# 结束原因。**每个出口都要有一个**，否则 trace 里会看到"跑完了但不知道为什么停"。
FinishReason = Literal[
    "answered",          # 正常答完
    "refused",           # 按设计拒答（越界 / 缺信息）
    "guard_steps",       # 撞上最大步数
    "guard_repeat",      # 撞上重复调用限制
    "guard_no_progress",  # 撞上无进展限制
    "guard_budget",      # 撞上 token 预算
    "error",             # 兜底：出了没预料到的错
]


class InjectedArticle(BaseModel):
    """注入 prompt 的一条法条。**`index` 就是答案里 `[1]` 那个编号。**

    ⚠️ 这个编号是本项目"引用可校验"的地基，所以有两条硬规则：

    1. **编号从 1 开始且连续**。模型看到的上下文里，第 i 条前面就写 `[i]`，
       模型只要照抄编号就行——不需要它自己数、也不允许它自己编。
    2. **编号 ↔ article_key 的映射只在这里产生**，`citation_report` 校验时
       只查这张表。所以校验是**字符串查找**，不是语义理解，**零 LLM、零成本、可复现**。
       这是整个项目里唯一一个"完全不用模型就能算出来的回答质量指标"。
    """

    index: int
    article_key: str
    citation: str  # 《劳动合同法》第四十七条
    law: str
    article_no: str
    text: str
    score: float = 0.0
    source: str = "hit"  # hit=检索命中 / neighbor=邻条扩展带出来的


class ToolCallRecord(BaseModel):
    """一次工具调用。**成功和失败都记**——失败记录才是失败归因（F27）的原料。"""

    step: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: str = ""  # ok / degraded / failed
    attempts: int = 0
    elapsed_ms: float = 0.0
    idempotency_hit: bool = False
    breaker_state: str = "closed"
    error: dict[str, Any] | None = None
    payload: Any = None
    """结果（可 JSON 序列化）。失败时为 None。"""


class CitationReport(BaseModel):
    """引用校验结果（F25）。**全部字段都是程序算出来的，没有模型参与。**"""

    cited: list[int] = Field(default_factory=list)
    """答案里出现过的编号（去重、保序）。"""

    valid: list[int] = Field(default_factory=list)
    """编号存在且落在注入范围内 → 这些是**有效引用**。"""

    invalid: list[int] = Field(default_factory=list)
    """模型编出来的编号（注入里根本没有第 7 条，它却写了 `[7]`）。"""

    uncited: list[int] = Field(default_factory=list)
    """注入了但没被引用。**这个数不是越低越好**——注入了 8 条只引用 2 条很正常，
    但"注入了 8 条一条都没引用"就说明模型没在用检索结果，那是要报的。"""

    @property
    def precision(self) -> float:
        """引用有效率 = 有效引用 / 全部引用。**编造编号会让它掉下去。**"""
        total = len(self.cited)
        return len(self.valid) / total if total else 0.0

    @property
    def coverage(self) -> float:
        """引用覆盖率 = 有效引用 / 可引用条数。**反映"检索给了料但模型没用"。**"""
        if not self.cited and not self.uncited:
            return 0.0
        denom = len(self.valid) + len(self.uncited)
        return len(self.valid) / denom if denom else 0.0


class TraceStep(BaseModel):
    """一步的埋点。**这是"线上出问题怎么查"的第一手证据**（F28）。

    刻意记 `elapsed_ms` 和 `note`：出问题时看的不是"走了哪几步"，
    而是"哪一步慢、哪一步是降级走的"。
    """

    step: int
    node: str
    elapsed_ms: float = 0.0
    note: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class GuardHit(BaseModel):
    """撞上某道防死循环限制的现场记录。**留证据，不只是"掐了"。**"""

    limit: str
    message: str
    step: int
    detail: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """白板本体。

    `Annotated[..., operator.add]` 的两个字段是**累加字段**：
    LangGraph 在合并节点返回的增量时用 `+` 而不是覆盖。
    这就是为什么节点里可以 `return {"trace": [一步]}` 却不会把之前的 trace 冲掉。
    """

    # ---- 输入 ----
    question: str = ""
    history: list[dict[str, str]] = Field(default_factory=list)
    """多轮对话历史，每项形如 `{"role": "user"/"assistant", "content": "..."}`。"""

    # ---- 中间产物 ----
    route: Route | None = None
    route_reason: str = ""
    """为什么这么路由。**必须记**——路由错了要能一眼看出是哪条规则判错的。"""

    injected: list[InjectedArticle] = Field(default_factory=list)
    retrieval_note: str = ""
    retrieved_keys: list[str] = Field(default_factory=list)
    """本轮检索命中的条级键。给评测层算 Recall 用，也给防死循环判"有没有进展"。"""

    tool_results: Annotated[list[ToolCallRecord], operator.add] = Field(default_factory=list)
    draft: str = ""
    """生成节点的原始输出（**未经过引用校验**）。"""

    tool_messages: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    """Function Calling 往返的原始消息（assistant 的 `tool_calls` + tool 的返回）。

    **必须原样留着**：第二轮工具调用时，模型需要看到第一轮的调用和结果，
    否则它会重复调一遍已经调过的工具。这也是"模型为什么会原地打转"的一个
    真实成因——如果消息没带全，模型会以为还没查过。
    """

    next_step: str = ""
    """`guard` 节点写给条件边的"下一步去哪"。

    为什么要有这个字段：**让所有闸门判定集中在一个节点里**。
    否则每个工作节点后面都要挂一个条件边、各自判一次，四道闸门的逻辑
    会散成四份。代价是这个字段看着有点怪（状态里存了个"下一步"），
    换来的是"防死循环只有一处实现"——值这个价。
    """

    forced_final: bool = False
    """已经强行收尾过一次了。

    ⚠️ 这个字段是**防死循环闸门自身死循环**用的，少了它图会转不出来：

    撞闸门时我们不想直接结束——手里已经有工具结果和检索结果了，
    白白丢掉太可惜（用户会收到一个空白回复）。所以闸门的做法是
    "**不再允许新的工具调用，但允许再生成一次答案**"。
    而"再生成一次"意味着 `guard → generate → guard` 会再撞一次同样的闸门，
    于是又派发一次生成……无限循环。

    `forced_final=True` 把这条路堵死：**强行收尾只能有一次**。
    """

    # ---- 记账 ----
    steps: int = 0
    tokens_used: int = 0
    trace: Annotated[list[TraceStep], operator.add] = Field(default_factory=list)
    progress_keys: Annotated[list[str], operator.add] = Field(default_factory=list)
    """每一步"产出了什么"的指纹。`guard.py` 靠它判有没有进展。

    累加字段（`operator.add`），所以节点只要 `return {"progress_keys": ["工具:..."]}`
    就会**追加**，不会把前面几步的痕迹冲掉。

    ⚠️ **判"没进展"的读法不是"数末尾连续相同"**（那是第一版的做法，会漏掉
    "换个说法来回问同一件事"这种最常见的原地打转）。真正的读法是
    "末尾连续几步产出的东西以前出现过"——见 `guard.no_progress_run()`。
    """

    guard_hits: Annotated[list[GuardHit], operator.add] = Field(default_factory=list)

    # ---- 输出 ----
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    """答案最终引用的法条（`lcl#47` 这种键），**已过滤掉无效引用**。"""

    citation_report: CitationReport | None = None
    refusal: dict[str, Any] | None = None
    finish_reason: FinishReason | None = None
    finish_detail: str = ""

    # ---- 诊断 ----
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    invariant_violations: Annotated[list[str], operator.add] = Field(default_factory=list)
    """被违反的断言（F18）。dev 模式下这会直接抛异常，走不到这里；
    prod 模式下攒在这里，随响应一起回给前端，不静默吞掉。"""

    # ---- 便于调试 ----
    def injected_by_index(self) -> dict[int, InjectedArticle]:
        return {a.index: a for a in self.injected}

    def seen_state(self) -> str:
        """给测试用的一行摘要：`route=law steps=3 injected=8 cited=2 [1,2]`。"""
        cited = self.citation_report.cited if self.citation_report else []
        return (
            f"route={self.route} steps={self.steps} injected={len(self.injected)} "
            f"cited={len(cited)} {cited}"
        )


__all__ = [
    "AgentState",
    "CitationReport",
    "FinishReason",
    "GuardHit",
    "InjectedArticle",
    "Route",
    "ToolCallRecord",
    "TraceStep",
]
