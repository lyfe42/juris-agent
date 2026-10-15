"""路由：判断用户这句话该走哪条路（F14）。

四种去向（词表与判定顺序见 `configs/router.yaml`）：

    law           法条问答  → 检索 → 生成
    calc          计算题    → 调计算工具 → 生成
    out_of_scope  越界问题  → 拒答
    chitchat      闲聊      → 直接回一句

## 这是"规则路由"，不是"模型路由"——理由在下面的注释里，面试会问

先说结论：**这一步刻意不让大模型判断。** 三个理由按重要性排：

1. **路由错了，后面全白跑。** 把一道加班费的题判成"越界"直接拒答，
   后面的检索、精排、生成再准也没用——错误发生在最上游，而且是**不可挽回**的。
   最不可挽回的一步，最不该押在"模型这次恰好判对了"上。
2. **免费、零延迟、可复现。** 规则匹配是微秒级的字符串包含，不花钱、不联网、
   同样的输入永远给同样的输出。而 LLM 路由要 300~800ms、要钱，
   而且（本项目实测）`temperature=0` 也不保证可复现。
3. **错了能解释、能修。** 规则路由的输出是"因为句子里有『离婚』、且没有任何
   劳动法信号，所以判越界"——这句话能进日志、能被用户当场纠正、能变成
   配置文件里加一个词。LLM 路由只能给你一个标签，你不知道它为什么这么判。

**但要说清楚它接不住什么**（面试官若追问，答这段，比硬吹"规则路由更好"强）：
- **指代**："那我要是不签呢？"——句子里没有任何关键词，规则路由只能看到"不签"。
  本文件靠对话历史补全（`_context_text`），但补全得很粗糙：它把上一轮的问题
  一起拿来匹配词表。真正稳的做法是把上一轮的**槽位**（用户问的是哪部法、哪一条、
  哪个 계산器）继承下来，那是记忆层（F19）的活。
- **反讽和口语**："公司这波操作真是绝了，说裁就裁"——没有"裁员"两个字但有"裁"，
  规则接不住。这类只能靠 badcase 回流慢慢加词。

## 一个容易被忽略的细节：判定顺序决定了谁会误判

`in_scope` 必须在 `out_of_scope` **之前**判。否则"我离婚了，公司把我辞了，
能拿多少补偿"会因为句子里有「离婚」而被判成越界——**一道标准的劳动法题被扔掉**。
代码里靠 `_decide` 里的注释把这条顺序钉住，改的时候别调换。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from pydantic import BaseModel, Field

from juris_agent.agent.state import Route
from juris_agent.config import get_settings


class RouteDecision(BaseModel):
    """路由结果。**带 `reason` 和 `matched` 是刻意的**——见模块 docstring 第 3 条理由。"""

    route: Route
    reason: str
    matched: list[str] = Field(default_factory=list)
    """命中的词。前端和日志直接展示这个，用户能立刻判断"是不是判错了"。"""

    calc_kind: str | None = None
    """计算题的细分类型（`计算` / `试用期` / `时效` / `加班` / `合同体检`）。
    计算节点靠它决定去调哪个工具。"""


@dataclass(frozen=True)
class RouterRules:
    """`configs/router.yaml` 的内存形态。**只读**——它被所有请求共享。"""

    in_scope: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    chitchat: tuple[str, ...]
    calc: Mapping[str, tuple[str, ...]]
    """单关键词信号：命中**任一**即算。适合"这个词只可能出现在这一类问题里"的情况。"""

    calc_all_of: Mapping[str, tuple[tuple[str, ...], ...]]
    """组合信号：**每一组都要命中一组**才算。见 `configs/router.yaml` 的说明。

    形态是 `类型 → ((组1的词...), (组2的词...))`，
    判定是 `all(any(w in text for w in group) for group in groups)`。
    """

    calc_priority: tuple[str, ...]
    chitchat_max_chars: int = 12
    calc_min_chars: int = 4

    @classmethod
    def load(cls, path: Path | None = None) -> "RouterRules":
        p = path or (get_settings().paths.configs / "router.yaml")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        calc_raw = raw.get("calc_signals") or {}
        all_of_raw = raw.get("calc_all_of") or {}
        # 优先级列表缺省时，用"单关键词组 + 组合组"的全部键，
        # 顺序取 YAML 里的书写顺序（Python 3.7+ 的 dict 保序）
        default_priority = tuple(calc_raw) + tuple(k for k in all_of_raw if k not in calc_raw)
        return cls(
            in_scope=tuple(raw.get("in_scope_signals") or ()),
            out_of_scope=tuple(raw.get("out_of_scope_signals") or ()),
            chitchat=tuple(raw.get("chitchat_signals") or ()),
            calc={k: tuple(v or ()) for k, v in calc_raw.items()},
            calc_all_of={
                k: tuple(tuple(g or ()) for g in groups or ()) for k, groups in all_of_raw.items()
            },
            calc_priority=tuple(raw.get("calc_priority") or default_priority),
            chitchat_max_chars=int(raw.get("chitchat_max_chars", 12)),
            calc_min_chars=int(raw.get("calc_min_chars", 4)),
        )


def _hits(text: str, words: Sequence[str]) -> list[str]:
    """返回命中的词，**长词优先**。

    长词优先是为了让日志好看也好解释：句子里有「劳动合同」，报 `劳动合同`
    比报 `合同` 有用得多（后者会让人以为判据是"合同"这个泛词）。
    """
    found = [w for w in words if w and w in text]
    # 去重后按长度降序：长词更具体，先报它
    return sorted(dict.fromkeys(found), key=len, reverse=True)


def _context_text(question: str, history: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    """返回 `(用于判定的文本, 说明)`。

    **文本 = 本轮问题 + 最近一轮用户发言**（如果有）。

    为什么要把历史拼进来：多轮对话里用户会说"那第二种呢""我不签会怎样"，
    这些句子单独看没有任何关键词，但接在上一轮后面就很好判。
    拼接是**很粗**的补全——它不区分"上一轮是谁说的"，也不管隔了几轮。
    真正的做法是继承槽位（F19），这里先做到"能接住最简单的追问"。

    ⚠️ 拼进来的部分最多 200 字：历史太长会让"上一轮里有『离婚』"污染这一轮的判定。
    这是个**权衡**，不是随便定的数——短了接不住追问，长了误判变多。
    """
    if not history:
        return question, "仅本轮问题"
    prev_user = ""
    for m in reversed(history):
        if str(m.get("role", "")) == "user":
            prev_user = str(m.get("content", ""))
            break
    if not prev_user:
        return question, "仅本轮问题"
    merged = f"{prev_user[-200:]} {question}"
    return merged, "本轮问题 + 上一轮用户发言（接追问用）"


def decide(
    question: str,
    *,
    history: Sequence[Mapping[str, str]] = (),
    rules: RouterRules | None = None,
) -> RouteDecision:
    """判定这一步该走哪条路。**纯函数，无副作用，可以放心单测。**"""
    r = rules or RouterRules.load()
    # 短句才拼接历史。长句本身就是完整的，拼上历史只会引入噪声。
    if len(question.strip()) <= r.chitchat_max_chars + 8:
        text, how = _context_text(question, history)
    else:
        text, how = question, "仅本轮问题（句子已足够完整）"

    in_scope_hits = _hits(text, r.in_scope)
    out_hits = _hits(text, r.out_of_scope)

    # ① 有劳动法信号 → 一律算在范围内。**这一步必须在越界之前**（见模块 docstring）。
    if in_scope_hits:
        kind, calc_hits = _match_calc(question, r)
        if kind and len(question.strip()) >= r.calc_min_chars:
            return RouteDecision(
                route="calc",
                calc_kind=kind,
                matched=in_scope_hits[:6] + calc_hits[:4],
                reason=f"命中劳动法名词{in_scope_hits[:3]}，且命中计算意图「{kind}」→ 走计算",
            )
        return RouteDecision(
            route="law",
            matched=in_scope_hits[:8],
            reason=f"命中劳动法名词{in_scope_hits[:3]} → 走法条问答（{how}）",
        )

    # ② 没有劳动法信号，但有越界信号 → 拒答
    if out_hits:
        return RouteDecision(
            route="out_of_scope",
            matched=out_hits[:8],
            reason=f"命中「{out_hits[0]}」等非劳动法词汇，且无任何劳动法名词 → 拒答",
        )

    # ③ 闲聊：**必须同时满足"命中问候语"和"句子够短"**
    stripped = question.strip()
    chat_hits = _hits(stripped, r.chitchat)
    if chat_hits and len(stripped) <= r.chitchat_max_chars:
        return RouteDecision(
            route="chitchat",
            matched=chat_hits,
            reason=f"命中问候语「{chat_hits[0]}」且仅 {len(stripped)} 字（≤{r.chitchat_max_chars}）→ 闲聊",
        )

    # ④ 劳动法信号和越界信号都没有，但**组合信号**可能认得出来。
    #    "帮我看看这份合同有没有问题"就是这一类：它一个劳动法名词都没有，
    #    但"合同"+"问题"这个组合明确指向合同体检。
    kind, calc_hits = _match_calc(question, r, allow_all_of=True)
    if kind and len(question.strip()) >= r.calc_min_chars:
        return RouteDecision(
            route="calc",
            calc_kind=kind,
            matched=calc_hits[:8],
            reason=f"命中组合信号「{kind}」→ 走计算/工具",
        )

    # ⑤ 兜底成法条问答。**不兜底成拒答**——
    #    答错了用户能看出来（有引用、可点开原文），拒错了用户直接就走了。
    return RouteDecision(
        route="law",
        matched=[],
        reason="未命中任何信号词，按法条问答兜底（宁可白检索一次也不直接拒答）",
    )


def _match_calc(
    question: str, r: RouterRules, *, allow_all_of: bool = False
) -> tuple[str | None, list[str]]:
    """判断是不是计算题，是的话返回细分类型。

    按 `calc_priority` 的顺序匹配——**越具体的信号优先**。
    一句话里同时有「试用期」和「多少钱」时，判成试用期而不是"计算"，
    因为前者能直接对应到 `check_probation` 这个工具。

    `allow_all_of` 控制**组合信号能不能用**：

    - 有劳动法信号时（`decide` 走①）：`False`。此时句子本身已经足够明确，
      再用组合信号只会让"劳动合同签了但公司不给钱"被误判成合同体检。
    - 一点信号都没有时（`decide` 走④）：`True`。这是组合信号唯一的用武之地——
      那类句子里一个专业名词都没有，只能靠"词和词一起出现"来认。
    """
    for kind in r.calc_priority:
        words = r.calc.get(kind)
        if words:
            hits = _hits(question, words)
            if hits:
                return kind, hits
        if not allow_all_of:
            continue
        groups = r.calc_all_of.get(kind)
        if not groups:
            continue
        matched: list[str] = []
        ok = True
        for group in groups:
            g_hits = _hits(question, group)
            if not g_hits:
                ok = False
                break
            matched.extend(g_hits)
        if ok and matched:
            return kind, matched
    return None, []


def calc_tool_for(kind: str | None) -> str | None:
    """计算类型 → 工具名。**这张表是路由和工具层唯一的接触点。**

    新增计算器时要同时改两个地方：`configs/router.yaml` 的 `calc_signals`
    加一组词，这张表加一行。`tests/test_router.py` 里有一条断言盯着"这张表的每个
    取值都是真实注册的工具名"，就是为了防止只改一半。
    """
    return {
        "计算": "calc_severance",       # 最泛的"能赔多少" → 默认猜经济补偿
        "试用期": "check_probation",
        "时效": "calc_limitation",
        "加班": "calc_overtime",
        "合同体检": "review_contract",
    }.get(kind or "")


__all__ = ["RouteDecision", "RouterRules", "calc_tool_for", "decide"]
