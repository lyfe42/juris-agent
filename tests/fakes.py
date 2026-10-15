"""测试用的假件。**不含任何真实网络调用、不加载任何模型。**

## 为什么 Agent 层的测试必须要假件

真实的 Agent 测试有三个不可接受的属性：

1. **要钱**（每次调用都花钱）→ 跑一次 CI 就烧掉几块钱，于是没人愿意跑
2. **不可复现**（模型有随机性）→ 测出来的红绿可能是运气
3. **慢**（一次问答十几秒）→ 一套测试跑十分钟，就没人跑了

而本项目要测的东西——**路由判得对不对、闸门拦不拦得住、引用校验算得准不准**——
**全都不需要真模型**。它们要么是纯函数（路由、校验），要么是"给定模型输出、
链路是否正确"（图）。假件恰好是测这些的最合适工具：

> **真模型测"效果"，假件测"逻辑"。** 效果那部分归评测层（`reports/`），
> 逻辑那部分归单测（`tests/`）。**别用单测测效果**——那会把两个东西都做坏：
> 单测变得又慢又脆，而效果其实没被量化。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from juris_agent.llm import LLMResponse, Usage

# 两条固定的假法条。键用真实的 `lcl#47` / `lcl#46`，
# 因为 `article_key` 的形状（`法id#条号`）会出现在断言里，用真形状更稳。
FAKE_ARTICLES: list[dict[str, Any]] = [
    {
        "article_key": "lcl#47",
        "law": "劳动合同法",
        "article_no": "第四十七条",
        "text": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。"
        "六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。",
        "score": 0.5,
        "citation": "《劳动合同法》第四十七条",
        "source": "hit",
    },
    {
        "article_key": "lcl#46",
        "law": "劳动合同法",
        "article_no": "第四十六条",
        "text": "有下列情形之一的，用人单位应当向劳动者支付经济补偿：……",
        "score": 0.4,
        "citation": "《劳动合同法》第四十六条",
        "source": "hit",
    },
]


class FakeRetriever:
    """假检索器。`search()` 恒返回 `FAKE_ARTICLES`（可按 `top_k` 截断）。

    可配置项只有两个，都是为了让测试能构造**边界**而不是"正常情况"：

    - `hits`：返回什么。给空列表就能测"检索为空 → 应该拒答"这条路径。
    - `calls`：记录被查过几次、查的什么。用来断言"检索只发生一次"这类结构性质。
    """

    def __init__(self, hits: Iterable[dict[str, Any]] | None = None) -> None:
        self.hits = list(FAKE_ARTICLES if hits is None else hits)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]

    def describe_stages(self) -> str:
        return "fake 1ms"


class ScriptedLLM:
    """按剧本回话的假模型。

    `script` 是一个列表，每次 `chat()` 弹出第一个作为本次回复。
    剧本用完之后的默认回复是 `{"text": ""}`——**故意给空**，
    因为"剧本没写够"是一种真实的 bug（节点比预期多调了一次），
    让它在测试里表现为"答案是空的"比表现为"编了一段话"更容易发现。

    另外它记录每次调用的**消息条数**（`msg_counts`）和**是否带 tools**
    （`tool_flags`）。这两个是断言"工具循环有没有把历史带全"的关键证据——
    第二轮调用如果消息数和第一轮一样多，就说明工具往返的消息丢了。
    """

    def __init__(
        self,
        script: list[dict[str, Any]] | None = None,
        *,
        tail: dict[str, Any] | None = None,
    ) -> None:
        self.script = list(script or [])
        # 剧本用完之后的默认回复。**默认是空正文**——
        # 因为"剧本没写够"是一种真实的 bug（节点比预期多调了一次），
        # 让它在测试里表现为"答案是空的"比表现为"编了一段话"更容易发现。
        #
        # 但有些测试（比如防死循环）**故意不关心还有几轮**——它们只关心
        # "最后有没有停下来"。那种场景传 `tail=` 更清晰，比数着轮次写剧本稳。
        self.tail = tail if tail is not None else {"text": ""}
        self.msg_counts: list[int] = []
        self.tool_flags: list[bool] = []
        self.messages: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        self.msg_counts.append(len(messages))
        self.tool_flags.append(bool(kw.get("tools")))
        self.messages.append(list(messages))
        step = self.script.pop(0) if self.script else dict(self.tail)
        if isinstance(step, str):  # 允许直接写一段文本
            step = {"text": step}
        return LLMResponse(
            text=step.get("text", ""),
            model="fake",
            usage=Usage(total_tokens=step.get("tokens", 10)),
            tool_calls=step.get("tool_calls", []),
        )


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """造一条 Function Calling 的调用项（形状照抄 OpenAI 协议）。

    ⚠️ `arguments` 是**字符串**不是对象——协议就这么定的。
    测试里如果用对象，就会掩盖"代码忘了 `json.loads`"这个 bug。
    所以这个 helper 默认做 `json.dumps`，**要测"参数不是合法 JSON"
    就显式传 `raw=`**。
    """
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def raw_tool_call(call_id: str, name: str, raw: str) -> dict[str, Any]:
    """造一条**参数不是合法 JSON** 的调用。测"喂回错误让模型重试"那条路径用。"""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


__all__ = [
    "FAKE_ARTICLES",
    "FakeRetriever",
    "ScriptedLLM",
    "raw_tool_call",
    "tool_call",
]
