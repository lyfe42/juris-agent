"""把图里发生的事翻译成给前端的事件（F29）。

## 这个文件是干什么的：两层粒度的翻译

一次提问在系统里会产生**两种完全不同粒度**的信息，前端两种都要：

| 粒度 | 来源 | 例子 | 为什么需要 |
|---|---|---|---|
| **按节点分步** | LangGraph 的 `stream_mode="updates"` | 路由判定、检索到 8 条、工具失败 | 让用户看到"它在干什么"，不是黑箱 |
| **逐字** | LLM 客户端的 `on_delta` 回调 | "试用期" / "最长" / "六个月。" | 让用户**立刻**看到字在出来 |

⚠️ **两者互不替代，少一个都会明显掉档次**：

- 只有分步：用户点完发送，盯着转圈等 7 秒，然后答案整段蹦出来。
- 只有逐字：答案慢慢出来，但中间那 7 秒的检索/工具/路由完全不可见——
  而"可见"恰恰是这个项目要展示的东西（面试官要直接看见检索和工具）。

所以两种事件都推，前端按自己的节奏渲染。

## 事件名为什么是这几个

名字照抄"系统里真实发生的事"，不发明新词。列全在这里，
前端和后端都以此为准（**不在别处定义第二份**）：

| 事件 | 什么时候 | 前端拿它干什么 |
|---|---|---|
| `start` | 请求被接受 | 建一个空的这一轮 |
| `memory` | 记忆层组装完 | 显示"记得 4 项事实"，可展开看出处 |
| `route` | 路由判完 | 显示"这题走计算路"及理由 |
| `retrieval` | 检索完 | 显示命中条数、耗时、可点开原文 |
| `tool_call` | 要调工具了 | 显示"正在算经济补偿" |
| `tool_result` | 工具返回 | 显示结果 / 失败原因 / 是不是幂等命中 |
| `missing` | 有必需事实没给 | 显示"还需要：月工资" |
| `thinking` | 推理模型边想边吐 | 显示一行"思考中…"（**不显示内容**）|
| `answer` | 答案的一段 | 追加到答案区 |
| `citations` | 引用校验完 | 显示"引用了 2 条 / 编造了 1 条（已删）" |
| `guard` | 撞上防死循环闸门 | 显示"因为反复调用同一个工具，已收尾" |
| `refusal` | 拒答 | 显示拒答卡片（类别 + 原因 + 建议）|
| `done` | 全部结束 | 用**权威的最终状态**覆盖答案区，显示成本/trace |
| `error` | 出错了 | 显示错误 |

⚠️ `done` 为什么必须**覆盖**答案区而不是"就那样"：
逐字流出来的字是模型的原始输出，而最终答案在 `generate` 里
**被引用校验处理过**（编造的编号会被删掉）。所以两者可能不一样。
**在 `done` 时用最终状态覆盖，是唯一能保证"屏幕上是什么、存下来就是什么"的做法。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from juris_agent.observability import get_logger

log = get_logger(__name__)


@dataclass
class Event:
    """一条要发给前端的事件。`name` 对应 SSE 的 `event:` 字段。"""

    name: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """渲染成 SSE 的一帧。

        ⚠️ **`ensure_ascii=False`**：默认（True）会把中文转成
        `\\u8bd5\\u7528\\u671f`，前端虽然也能正确显示，但
        **`curl` 出来的是一堆转义码**——而"curl 能看到事件按序流出"
        正是这个接口的验收标准。看起来能跑但没法演示，等于没做。

        ⚠️ 末尾必须有**两个**换行。一个换行只是这一帧的结束，
        两个换行才是"这一帧发完了"。少一个的话，SSE 客户端会一直等下一帧，
        表现为"事件全都收到了，但界面一直不更新"。
        """
        import json

        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.name}\ndata: {payload}\n\n"


def as_dict(obj: Any) -> dict[str, Any]:
    """pydantic 对象和普通 dict 都能吃。

    为什么要这个：LangGraph 在合并状态增量时**不保证**把 pydantic 对象
    原样留着（它会走一遍 schema 校验，类型对得上就留着，对不上就转）。
    写死 `obj.model_dump()` 会在某次 langgraph 升级后突然报
    `AttributeError: 'dict' object has no attribute 'model_dump'`——
    而那种错发生在生产路径上，测试未必盖得住。**两种都接受，成本 3 行。**
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {"value": obj}


def events_from_update(node: str, delta: dict[str, Any]) -> list[Event]:
    """一个节点的增量 → 若干事件。

    ⚠️ 这个函数**只读不写**，而且**不改 delta**。
    节点返回的增量是 LangGraph 要拿去合并的，动它会影响真正的状态
    （表现为"某一轮的记忆块莫名少了几个字段"）。
    """
    out: list[Event] = []
    d = {k: v for k, v in delta.items() if v is not None}

    if node == "remember":
        out.extend(_from_remember(d))
    elif node == "entry":
        out.append(
            Event(
                "route",
                {"route": d.get("route", ""), "reason": d.get("route_reason", "")},
            )
        )
    elif node == "retrieve":
        out.extend(_from_retrieve(d))
    elif node == "tool":
        out.extend(_from_tool(d))
    elif node == "generate":
        out.extend(_from_generate(d))
    elif node == "guard":
        out.extend(_from_guard(d))
    elif node == "refuse":
        out.append(Event("refusal", dict(d.get("refusal") or {})))
    elif node == "chitchat":
        # 闲聊不经过模型生成，所以没有逐字流——**整段推一次**。
        # 前端不必区分：它收到的都是 `answer` 事件，只是这次只有一段。
        if text := d.get("answer"):
            out.append(Event("answer", {"text": text}))

    # 任何节点写进 `errors` 的都要露出来——**被吞掉的错误是最贵的错误**。
    for e in d.get("errors") or []:
        out.append(Event("error", {"message": str(e), "node": node}))

    return out


def _from_remember(d: dict[str, Any]) -> list[Event]:
    """记忆层。**没有 `memory_block` 就什么都不发**——
    记忆层被关掉时（`JURIS_AGENT_MEMORY_ENABLED=false`）它只写一条 trace，
    这时发一个空事件会让前端显示"记得 0 项"，看起来像是记忆坏了。"""
    if not d.get("memory_policy") and not d.get("memory_block"):
        return []
    return [
        Event(
            "memory",
            {
                "policy": d.get("memory_policy", ""),
                "tokens": d.get("memory_tokens", 0),
                "layers": d.get("memory_layers", []),
                "slots": slots_from_provenance(d.get("slots_provenance") or {}),
                "dropped": d.get("memory_dropped", []),
                "missing": d.get("missing_slots", []),
            },
        )
    ]


def slots_from_provenance(provenance: dict[str, str]) -> list[dict[str, Any]]:
    """把槽位的出处字符串翻译成结构化的一条。

    输入的字符串形态是 `月工资: 12000（第1轮：『我月薪12000』）`
    （由 `memory/slots.py` 生成）。这里**不重新格式化它**，
    只做拆分——两份格式代码迟早会不一致，而"显示的和记的不一样"
    在没有原文对照的时候根本发现不了。

    拆不出来（格式变了）就**原样放进 `value`**，前端会把它当一句话显示。
    这比让字段变成空值好：**宁可显示得难看，不要显示得像是没有。**
    """
    out: list[dict[str, Any]] = []
    for label, raw in provenance.items():
        text = str(raw)
        value, turn, quote = text, 0, ""
        if "（第" in text:
            head, _, tail = text.partition("（第")
            value = head.strip()
            num = tail.split("轮", 1)[0]
            turn = int(num) if num.isdigit() else 0
            if "『" in tail and "』" in tail:
                quote = tail.split("『", 1)[1].split("』", 1)[0]
        out.append({"label": label, "value": value, "turn": turn, "quote": quote})
    return out


def _from_retrieve(d: dict[str, Any]) -> list[Event]:
    injected = [as_dict(a) for a in d.get("injected") or []]
    if not injected:
        # 检索为空会转拒答，`retrieve` 节点自己不写 refusal——
        # 所以这里只发一个 0 条的检索事件，拒答事件由 `refuse` 节点发。
        return [Event("retrieval", {"count": 0, "articles": [], "note": d.get("retrieval_note", "")})]
    return [
        Event(
            "retrieval",
            {
                "count": len(injected),
                "note": d.get("retrieval_note", ""),
                "articles": [
                    {
                        "index": a.get("index"),
                        "article_key": a.get("article_key", ""),
                        "citation": a.get("citation", ""),
                        # ⚠️ 正文**一起带上**。不带的话前端点开一条法条
                        # 要再发一次请求，而那次往返的延迟会让"点开看原文"
                        # 这个动作明显发顿。8 条正文才 1~2KB。
                        "text": a.get("text", ""),
                        "source": a.get("source", "hit"),
                        "score": a.get("score", 0.0),
                    }
                    for a in injected
                ],
            },
        )
    ]


def _from_tool(d: dict[str, Any]) -> list[Event]:
    out: list[Event] = []
    for rec in d.get("tool_results") or []:
        r = as_dict(rec)
        out.append(
            Event(
                "tool_call",
                {"tool": r.get("tool", ""), "args": r.get("args", {}), "step": r.get("step")},
            )
        )
        out.append(
            Event(
                "tool_result",
                {
                    "tool": r.get("tool", ""),
                    "status": r.get("status", ""),
                    "attempts": r.get("attempts", 0),
                    "elapsed_ms": r.get("elapsed_ms", 0.0),
                    "idempotency_hit": r.get("idempotency_hit", False),
                    "breaker_state": r.get("breaker_state", "closed"),
                    "payload": r.get("payload"),
                    "error": r.get("error"),
                    # 降级说明。**必须一路带到界面上**——「⚠️ 降级」这四个字
                    # 不说明任何事，用户需要知道"还能用，但精度下降了"，
                    # 否则他会把降级结果当正常结果用。
                    "warnings": list(r.get("warnings") or []),
                },
            )
        )
    if missing := d.get("missing_slots"):
        out.append(Event("missing", {"labels": list(missing)}))
    return out


def _from_generate(d: dict[str, Any]) -> list[Event]:
    report = as_dict(d.get("citation_report"))
    if not report:
        return []
    # ⚠️ 这里**不发 `answer`**。答案的正文是靠逐字流（`on_delta`）推的，
    # 在这里再发一次会让前端的答案区出现两份完全相同的回答。
    # `done` 事件会用最终状态里的答案覆盖一次——那才是权威版本。
    return [
        Event(
            "citations",
            {
                "cited": report.get("cited", []),
                "valid": report.get("valid", []),
                "invalid": report.get("invalid", []),
                "uncited": report.get("uncited", []),
            },
        )
    ]


def _from_guard(d: dict[str, Any]) -> list[Event]:
    out: list[Event] = []
    for hit in d.get("guard_hits") or []:
        h = as_dict(hit)
        out.append(
            Event(
                "guard",
                {"limit": h.get("limit", ""), "message": h.get("message", ""), "step": h.get("step")},
            )
        )
    return out


__all__ = ["Event", "as_dict", "events_from_update", "slots_from_provenance"]
