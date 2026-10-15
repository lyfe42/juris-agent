"""节点级断言：同一套检查，开发期抛异常、生产期软降级（F18）。

## 这个模块回答的是"你怎么保证它不出错"

面试官问这句时，最差的回答是"我写得很仔细"。好的回答是一个**机制**：

> 每个节点开头检查"我拿到的输入是不是我要的形态"，结尾检查"我产出的东西
> 是不是合格的"。检查的代码只有一份，但**行为随环境变**——

| 环境 | 违反断言时 | 为什么 |
|---|---|---|
| `JURIS_AGENT_ENV=dev` | **直接抛异常，整个请求 500** | 开发期就要它炸。偷偷降级跑过去，你会以为测过了 |
| `JURIS_AGENT_ENV=prod` | **记进 `state.invariant_violations`，继续跑** | 线上一个断言不该让用户拿不到答案 |

**"开发期炸、生产期软"不是双标，是因为两种环境的最优失败方式不同**：
开发期最贵的是"没发现 bug"，生产期最贵的是"用户用不了"。同一套断言、
两种失败策略，是同时满足这两个目标的最简做法。

## 断言的选法：只查"程序能判对错"的事

⚠️ 断言的覆盖面**故意很窄**。它只查**能用代码判对错**的东西：

- ✅ "答案里的 `[7]` 不存在于注入的 8 条里" —— 查字符串，确定对错
- ✅ "进生成节点时注入是空的" —— 查长度，确定对错
- ❌ "答案的法律结论对不对" —— **不查**。那需要律师，查不了。
  它属于 LLM-as-Judge（F26）的范围，而且要给"打分不准"留出被质疑的余地。

把能查的和不能查的混在一起，会让"断言全过"变成一句空话——
面试官问"断言过了就代表答案对吗"，你答"不代表，只代表结构合法"，
**这个回答本身就是加分项**。

## ⚠️ 生产期"软降级"不等于"静默吞掉"

违反断言时会 `logger.warning(...)` 并且把记录塞进
`state.invariant_violations`，最终**随响应一起回给前端**（`/chat` 的
`invariant_violations` 字段）。理由：断言的价值在于"被看见"，
一个不产生任何可见后果的检查，等于没写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from juris_agent.agent.state import AgentState

# 断言编号。**写进日志和响应的就是这个字符串**，改了会破坏历史日志的可检索性，
# 所以编号只增不改。
INV_GENERATE_NEEDS_CONTEXT = "INV-01"
INV_CITATION_INDEX_RANGE = "INV-02"
INV_CITATION_KEY_INJECTED = "INV-03"
INV_STEPS_WITHIN_BUDGET = "INV-04"
INV_ANSWER_NONEMPTY = "INV-05"
INV_REFUSAL_SHAPE = "INV-06"
INV_TOOL_PAYLOAD_SERIALIZABLE = "INV-07"
INV_ROUTE_DECIDED = "INV-08"


class InvariantError(RuntimeError):
    """dev 模式下违反断言时抛出的异常。

    **继承 RuntimeError 而不是 Exception**：它是"程序状态不对"，
    不是"用户输入不对"，捕获的粒度不一样。
    """

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        self.code = code
        self.detail = detail or {}
        super().__init__(f"[{code}] {message}")


@dataclass
class Violation:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """日志和前端里显示的一行。"""
        extra = f" {self.detail}" if self.detail else ""
        return f"[{self.code}] {self.message}{extra}"


class InvariantChecker:
    """按环境决定"炸"还是"记"。**每个请求一个实例**（它带 logger 上下文）。

    为什么不做成模块级函数：它需要一个"当前是哪次请求"的上下文，
    否则线上出问题只知道"有个断言挂了"，不知道是哪个 request_id。
    那正是 F28 要解决的问题，这里先把钩子留出来。
    """

    def __init__(
        self,
        *,
        env: str = "dev",
        logger: Any | None = None,
        on_violation: Callable[[Violation], None] | None = None,
    ) -> None:
        self.env = env
        self.strict = env != "prod"
        self._logger = logger
        self._on_violation = on_violation
        self.violations: list[Violation] = []

    # -- 核心 ------------------------------------------------------------- #

    def require(
        self,
        ok: bool,
        code: str,
        message: str,
        **detail: Any,
    ) -> list[str]:
        """断言 `ok`。返回"要写进 state 的违规记录"（空列表 = 通过）。

        **返回列表而不是直接改 state**：断言函数不知道节点会怎么组织返回值，
        硬要它去改 state 就把两个关注点混在一起了。节点里写成：

            return {"invariant_violations": self.inv.require(...), ...}
        """
        if ok:
            return []
        v = Violation(code=code, message=message, detail=detail)
        self.violations.append(v)
        if self._logger is not None:
            self._logger.warning("invariant_violated", code=code, message=message, **detail)
        if self._on_violation is not None:
            self._on_violation(v)
        if self.strict:
            # dev/ci：炸。**注意这里是在记完之后才炸的**，
            # 所以异常信息里带着"记过一笔"的状态，堆栈和日志能对上。
            raise InvariantError(code, message, detail)
        return [v.render()]

    # -- 具名断言（每个写成一个方法，方便 grep 和单测） -------------------- #

    def generate_needs_context(self, state: AgentState) -> list[str]:
        """INV-01：进生成节点时，注入的上下文不能是空的。

        空上下文情况下让模型"凭记忆回答"是本项目最不能接受的行为——
        法律答案必须有出处。正确的走法是**跳去拒答节点**而不是生成。
        """
        return self.require(
            len(state.injected) > 0,
            INV_GENERATE_NEEDS_CONTEXT,
            "进入生成节点但注入上下文为空。应改走拒答节点（「没检索到依据」），"
            "而不是让模型凭记忆作答。",
            injected=len(state.injected),
            route=state.route,
        )

    def citation_index_in_range(self, state: AgentState, cited: list[int]) -> list[str]:
        """INV-02：答案里的编号必须落在注入范围内。

        ⚠️ **这条只用于 dev 期发现"模型在编编号"**，生产期不据此改写答案——
        编造的编号由 `verify.py` 直接删掉（那是修复，不是检查）。
        把两者分开：断言负责"让我知道这事在发生"，校验负责"让用户看不到它"。
        """
        valid_range = {a.index for a in state.injected}
        bad = [i for i in cited if i not in valid_range]
        return self.require(
            not bad,
            INV_CITATION_INDEX_RANGE,
            f"答案引用了不存在的编号 {bad}（注入范围 {sorted(valid_range)}）。"
            "模型在编造引用。",
            bad=bad,
            valid_range=sorted(valid_range),
        )

    def citation_keys_injected(self, state: AgentState) -> list[str]:
        """INV-03：最终引用的法条键必须都来自注入集合。

        这是"可溯源"这句话的**机器可验证形式**：答案里出现的每一部法条，
        都必须是这次检索真的捞上来的，不能是模型自己想起来加上的。
        """
        injected_keys = {a.article_key for a in state.injected}
        stray = [c for c in state.citations if c not in injected_keys]
        return self.require(
            not stray,
            INV_CITATION_KEY_INJECTED,
            f"引用了未注入的法条 {stray}。答案里的每条法条都必须能追溯回本次检索结果。",
            stray=stray,
            injected=sorted(injected_keys),
        )

    def steps_within_budget(self, state: AgentState, max_steps: int) -> list[str]:
        """INV-04：步数不能超。**防的是闸门自身失效**——
        如果闸门没拦住而步数冲过去了，这条会立刻暴露。

        ## ⚠️ 强行收尾那一步，上限放宽一格

        撞闸门之后图会**再生成一次**（"别把已经算出来的东西扔掉"，见 `guard` 节点），
        那一步也是要计数的。于是正常行为就会走到 `max_steps + 1`：
        第 12 步走完撞上闸门，第 13 步是强行收尾。

        **闸门和断言之间得有"协议"。** 如果这里不放宽，就会出现
        "系统完全按设计行事，但断言报了一条违规"——那种噪音的下场是
        有人把这条断言注释掉，于是真正的"闸门失效"也再没人看得见。
        放宽一格的代价是：真的多跑了一步的 bug 要 2 步才报得出来。
        这个交换是划算的——**因为闸门本身已经在第 12 步硬停了**，
        断言在这里是第二道保险，不是唯一一道。
        """
        allowed = max_steps + (1 if state.forced_final else 0)
        return self.require(
            state.steps <= allowed,
            INV_STEPS_WITHIN_BUDGET,
            f"已走 {state.steps} 步，超过上限 {allowed}"
            + ("（含强行收尾放宽的 1 步）" if state.forced_final else "")
            + "。防死循环闸门可能失效。",
            steps=state.steps,
            max_steps=allowed,
            forced_final=state.forced_final,
        )

    def answer_nonempty(self, state: AgentState) -> list[str]:
        """INV-05：正常答完就必须有答案。**防止"跑完了但什么都没给用户"。**"""
        if state.finish_reason != "answered":
            return []
        return self.require(
            bool(state.answer.strip()),
            INV_ANSWER_NONEMPTY,
            "finish_reason=answered 但答案是空的。用户会收到一个空白回复。",
            draft_len=len(state.draft),
        )

    def refusal_shape(self, state: AgentState) -> list[str]:
        """INV-06：拒答必须带齐三件东西。

        缺了 `suggestion`，用户被拒绝之后不知道该去问谁——那种拒答比硬答还伤。
        """
        if state.finish_reason != "refused":
            return []
        r = state.refusal or {}
        missing = [k for k in ("reason", "category", "suggestion") if not r.get(k)]
        return self.require(
            not missing,
            INV_REFUSAL_SHAPE,
            f"拒答缺少字段 {missing}。拒答必须给出理由、类别和替代建议。",
            missing=missing,
        )

    def tool_payload_serializable(self, payload: Any, tool: str) -> list[str]:
        """INV-07：工具结果必须是能 JSON 序列化的。

        理由很实际：工具结果要**喂回模型**（走 HTTP JSON）并且**推给前端**（走 SSE），
        里面夹一个 `Decimal` 或 `datetime` 就会在链路某处炸掉，
        而那个位置离出错点很远，排查成本极高。在源头查一次，问题立刻定位。
        """
        import json

        try:
            json.dumps(payload, ensure_ascii=False, default=str)
            ok = True
            why = ""
        except (TypeError, ValueError) as e:
            ok, why = False, str(e)
        return self.require(
            ok,
            INV_TOOL_PAYLOAD_SERIALIZABLE,
            f"工具 {tool} 的返回结果不能 JSON 序列化：{why}",
            tool=tool,
        )

    def route_decided(self, state: AgentState) -> list[str]:
        """INV-08：走到需要路由的下游节点时，路由必须已经定下来。

        这条防的是**图的接线错误**：如果哪天线画错了，某个节点跳过了路由节点，
        它会拿到 `route=None`。没有这条断言，症状会是"莫名其妙走了 law 分支"，
        而不是"短路了，接线错了"。
        """
        return self.require(
            state.route is not None,
            INV_ROUTE_DECIDED,
            "节点需要 route 但它是 None。图的条件边接线可能有误，某个节点被跳过了。",
        )


# 全部具名断言的登记表。**测试里会遍历它**，保证每个断言至少有两条测试
# （违反时 dev 抛异常 / prod 记一笔）。加断言时记得同步加测试，
# 否则 `tests/test_invariants.py` 里那条"数量对得上"的断言会红——
# 那是故意的，防止有断言写了却从没被触发过。
NAMED_CHECKS = (
    "generate_needs_context",
    "citation_index_in_range",
    "citation_keys_injected",
    "steps_within_budget",
    "answer_nonempty",
    "refusal_shape",
    "tool_payload_serializable",
    "route_decided",
)

__all__ = [
    "INV_ANSWER_NONEMPTY",
    "INV_CITATION_INDEX_RANGE",
    "INV_CITATION_KEY_INJECTED",
    "INV_GENERATE_NEEDS_CONTEXT",
    "INV_REFUSAL_SHAPE",
    "INV_ROUTE_DECIDED",
    "INV_STEPS_WITHIN_BUDGET",
    "INV_TOOL_PAYLOAD_SERIALIZABLE",
    "NAMED_CHECKS",
    "InvariantChecker",
    "InvariantError",
    "Violation",
]
