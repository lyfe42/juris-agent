"""图的各个节点：每一步"真正做一件事"的地方（F15）。

## 节点的写法有一条铁律：**只返回增量，不改别的字段**

```python
def retrieve(self, state):        # ← 只读 state
    ...
    return {"injected": [...], "steps": state.steps + 1}   # ← 只写自己产出的字段
```

理由有两个，第二个更重要：

1. LangGraph 要靠这个增量做状态合并（累加字段靠 `+`，普通字段靠覆盖）。
2. **它是"这一步到底干了什么"的唯一定义。** 一眼看返回值就知道这个节点
   产出了哪几样东西、有没有顺手改别的。如果节点可以直接改 `state`，
   出了问题你只能靠读函数体去找"它到底动了什么"，那在十几个节点的图里是灾难。

## 两条路，两种做法（这是本文件最该记住的设计决定）

| 路由 | 走法 | 为什么不一样 |
|---|---|---|
| `law`（法条问答） | 检索 → 生成 | **检索是确定性的一步。** 该不该查、查什么，路由已经定了；让模型来决定只是多花一次调用、多一个出错的地方 |
| `calc`（计算题） | 检索 → **工具循环** → 生成 | **必须让模型自主决定。** 调哪个计算器、参数填什么（工作几年、月薪多少、哪种解除情形），只有读完问题才知道。这一段是真正的 ReAct |

面试被问"你的 Agent 是不是就是把 RAG 包了一层"，答案就是上面这张表：
**法条路走确定性管道（能算的不用模型），计算路走工具循环（必须用模型的地方才用）。**
两条路的差别是有理由的，不是随手写的。

## 工具循环为什么用 Function Calling 而不是"让模型输出 JSON 再解析"

你在黑马教程里见到的 ReAct 提示词写法（"Thought/Action/Action Input"，
然后正则去抠 Action）在这里有个致命问题：**抠错了会静默执行一个错参数的计算**。
"工作 3 年"抠成"工作 30 年"，答案差十倍，而日志里看不出异常。

Function Calling 把参数放在**结构化的 JSON 字段**里，服务端保证它是合法 JSON，
解析失败是**显式报错**而不是静默错值。代价是必须用支持 tools 的模型
（DeepSeek 的 `deepseek-flash` 支持），收益是"参数错了会报错，不会算错"。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from juris_agent.agent import prompts
from juris_agent.agent.guard import args_hash, check, progress_signature
from juris_agent.agent.invariants import InvariantChecker
from juris_agent.agent.router import calc_tool_for, decide
from juris_agent.agent.state import (
    AgentState,
    InjectedArticle,
    ToolCallRecord,
    TraceStep,
)
from juris_agent.agent.verify import resolve_keys, strip_invalid, verify_citations
from juris_agent.config import Settings, get_settings


def _now() -> float:
    return time.perf_counter()


@dataclass
class AgentDeps:
    """节点需要的外部资源。**全部显式注入，方便测试里换成假的。**

    惰性：`llm` 和 `retriever` 都允许是 `None`。为 None 时相关节点会走
    "明确报错"的路径，而不是"假装做过了"。
    """

    settings: Settings = field(default_factory=get_settings)
    registry: Any = None          # ToolRegistry
    executor: Any = None          # ToolExecutor
    retriever: Any = None         # PipelineRetriever（有 search(query, top_k) 即可）
    llm: Any = None               # LLMClient（有 chat(messages, **kw) 即可）
    logger: Any = None

    def checker(self, state: AgentState) -> InvariantChecker:
        """每个请求一个新的断言器（它要带上 request 上下文）。"""
        return InvariantChecker(env=self.settings.env, logger=self.logger)


class Nodes:
    """节点的集合。**用类而不是散函数**，因为每个节点都要拿到同一份依赖。

    这样 `graph.py` 里只需要 `nodes = Nodes(deps)` 然后
    `g.add_node("retrieve", nodes.retrieve)`——绑定方法，签名正好是 `(state)`。
    """

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    # ------------------------------------------------------------------ #
    # 入口：路由
    # ------------------------------------------------------------------ #

    def entry(self, state: AgentState) -> dict[str, Any]:
        """决定这一轮走哪条路。**同时是"第几步"的起点。**

        ⚠️ 路由**不用模型**——理由在 `router.py` 的模块 docstring 里，三条：
        最不可挽回的一步不该押在模型上、规则免费且可复现、错了能解释能改。
        """
        t0 = _now()
        d = decide(state.question, history=state.history)
        return {
            "route": d.route,
            "route_reason": d.reason,
            "steps": state.steps + 1,
            # 路由本身不算"产出"，但要让闸门看到这一步是活的。
            # 用路由结果当指纹：同一句话路由两次得到同样的指纹 → 会被判成没进展。
            "progress_keys": [f"route:{d.route}:{d.calc_kind or '-'}"],
            "trace": [
                TraceStep(
                    step=state.steps + 1,
                    node="entry",
                    elapsed_ms=(_now() - t0) * 1000,
                    note=f"→ {d.route}",
                    detail={"reason": d.reason, "matched": d.matched, "calc_kind": d.calc_kind},
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 检索（两条路都要走）
    # ------------------------------------------------------------------ #

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        """把法条捞上来并编号。

        **两条路由都先走这一步**（law 和 calc），所以生成节点永远有依据可引。
        否则计算题的答案里全是"根据法律规定"而不带编号，引用校验也就算不出东西。
        """
        t0 = _now()
        inv = self.deps.checker(state)
        r = self.deps.retriever
        if r is None:
            return {
                "steps": state.steps + 1,
                "retrieval_note": "检索器未装配（retriever=None）",
                "errors": ["检索器未装配，无法检索法条"],
                "next_step": "refuse",
                "progress_keys": ["retrieve:none"],
                "trace": [
                    TraceStep(
                        step=state.steps + 1,
                        node="retrieve",
                        elapsed_ms=(_now() - t0) * 1000,
                        note="检索器缺失",
                    )
                ],
            }

        top_k = self.deps.settings.retrieval.inject_k
        hits = r.search(state.question, top_k=top_k)
        injected = [
            InjectedArticle(
                index=i,
                article_key=h["article_key"],
                citation=h.get("citation") or h["article_key"],
                law=h.get("law", ""),
                article_no=h.get("article_no", ""),
                text=h.get("text", ""),
                score=float(h.get("score", 0.0)),
                source=str(h.get("source", "hit")),
            )
            for i, h in enumerate(hits, start=1)
        ]
        note = r.describe_stages() if hasattr(r, "describe_stages") else ""
        return {
            "injected": injected,
            "retrieved_keys": [a.article_key for a in injected],
            "steps": state.steps + 1,
            "retrieval_note": note,
            "invariant_violations": inv.require(
                bool(injected),
                "INV-00-retrieve-empty",
                f"检索「{state.question}」返回 0 条。将转拒答。",
                query=state.question,
            ),
            # 下一步去哪，这里就把两条路分开：
            #   - 检索为空 → 拒答。**不要去调模型**——没有依据时它一定会编。
            #   - calc 路 → 先进工具循环（让模型挑计算器、填参数）。
            #   - law  路 → 直接生成，不再让模型决定"要不要查"（已经查过了）。
            "next_step": ("refuse" if not injected else ("tool" if state.route == "calc" else "generate")),
            "progress_keys": [progress_signature(new_keys=[a.article_key for a in injected])],
            "trace": [
                TraceStep(
                    step=state.steps + 1,
                    node="retrieve",
                    elapsed_ms=(_now() - t0) * 1000,
                    note=f"{len(injected)} 条",
                    detail={"stages": note},
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 工具循环（只走 calc 路）
    # ------------------------------------------------------------------ #

    def tool(self, state: AgentState) -> dict[str, Any]:
        """一轮工具调用：让模型选工具+填参数 → 执行 → 把结果喂回去。

        **一次只走一轮**，由 `guard` 决定要不要再来一轮。这么拆的理由：
        循环控制权归图，不归节点。节点里写 `while True` 的话，
        防死循环的四道闸门就得在这个循环里再实现一遍——**逻辑重复两份**，
        而且以后加闸门要记得改两个地方。
        """
        t0 = _now()
        llm = self.deps.llm
        inv = self.deps.checker(state)
        if llm is None:
            return {
                "steps": state.steps + 1,
                "errors": ["LLM 未装配，无法进行工具调用"],
                "next_step": "refuse",
                "refusal": prompts.build_refusal(
                    category="internal_error",
                    reason="服务未正确装配大模型客户端",
                    suggestion="请联系维护者；或改用「只查法条」的问法",
                ),
                "progress_keys": ["tool:no-llm"],
                "trace": [
                    TraceStep(step=state.steps + 1, node="tool", note="LLM 缺失", elapsed_ms=0.0)
                ],
            }

        messages = self._agent_messages(state)
        tools = self.deps.registry.to_openai_tools() if self.deps.registry else None

        try:
            resp = llm.chat(
                messages,
                tools=tools,
                temperature=0.0,
                max_tokens=self.deps.settings.tool_max_tokens,
            )
        except Exception as e:  # noqa: BLE001 — LLM 挂了不能把整个会话带走
            return {
                "steps": state.steps + 1,
                "errors": [f"工具循环中的模型调用失败：{type(e).__name__}: {e}"],
                "next_step": "generate",  # 退一步：用已有信息生成（可能是"依据不足"）
                "progress_keys": ["tool:llm-error"],
                "trace": [
                    TraceStep(
                        step=state.steps + 1,
                        node="tool",
                        elapsed_ms=(_now() - t0) * 1000,
                        note=f"模型调用失败 {type(e).__name__}",
                    )
                ],
            }

        tokens = state.tokens_used + (resp.usage.total_tokens or 0)
        base_detail = {"tool_calls": len(resp.tool_calls), "tokens": resp.usage.total_tokens}

        # 模型没调工具了 → 它认为自己掌握的信息够了 → 去生成答案。
        if not resp.wants_tools:
            return {
                "steps": state.steps + 1,
                "tokens_used": tokens,
                "next_step": "generate",
                "progress_keys": ["tool:none"],
                "trace": [
                    TraceStep(
                        step=state.steps + 1,
                        node="tool",
                        elapsed_ms=(_now() - t0) * 1000,
                        note="模型不再调用工具",
                        detail=base_detail,
                    )
                ],
            }

        # —— 执行工具 ——
        records: list[ToolCallRecord] = []
        tool_messages: list[dict[str, Any]] = []
        refused_by_model = False

        for call in resp.tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name", ""))
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                parse_error: str | None = None
            except (json.JSONDecodeError, TypeError) as e:
                # **参数不是合法 JSON 不当成致命错误。**
                # 把它变成一条"参数不合法"的工具反馈喂回模型让它重试——
                # 那比在这里猜它想传什么好（猜错会执行一个用户没要求的计算）。
                args = {}
                parse_error = f"参数不是合法 JSON：{e}。请重新给出合法 JSON 参数。"

            if parse_error:
                records.append(
                    ToolCallRecord(
                        step=state.steps + 1,
                        tool=name,
                        args={},
                        status="failed",
                        error={"error_type": "validation_error", "message": parse_error},
                    )
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(
                            {"error": parse_error, "retry": True}, ensure_ascii=False
                        ),
                    }
                )
                continue

            outcome = self.deps.executor.call(name, args, idempotency_key=args_hash(args))
            payload = outcome.payload()
            rec = ToolCallRecord(
                step=state.steps + 1,
                tool=name,
                args=args,
                status=outcome.status.value,
                attempts=outcome.attempts,
                elapsed_ms=outcome.elapsed_ms,
                idempotency_hit=outcome.idempotency_hit,
                breaker_state=outcome.breaker_state,
                error=outcome.error,
                payload=payload,
            )
            records.append(rec)

            if name == "refuse_to_answer" and outcome.ok:
                # 模型主动拒答。**这是被鼓励的**（见 prompts.AGENT_SYSTEM），
                # 所以要走正常的拒答收尾，而不是当成失败。
                refused_by_model = True

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(
                        payload if outcome.ok else {"error": outcome.error},
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

            # 出参可序列化断言（INV-07）：工具结果要喂回模型、还要推给前端，
            # 里面夹一个 Decimal 会在链路远处炸掉，很难查。
            inv.tool_payload_serializable(payload, name)

        # 助手消息也要存——下一轮模型必须看到"我调过什么、拿到什么"，
        # 否则它会重复调同一个工具（这是"原地打转"的一个真实成因）。
        assistant_msg = {
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": resp.tool_calls,
        }

        ok_records = [r for r in records if r.status == "ok"]

        if refused_by_model:
            refusal = next(
                (r.payload for r in records if r.tool == "refuse_to_answer" and r.payload), None
            ) or {}
            next_step = "refuse"
        elif ok_records:
            # 至少有一个工具成功 → 再给模型一轮机会去调别的工具（混合问题：
            # 先 search_law 查条文，再 calc_severance 算钱）。
            next_step = "tool"
        else:
            # 全挂了 → 不再给轮次。再给一轮它大概率会用同样的参数再试一次，
            # 那不叫重试，那叫烧钱（真正的重试已经在 executor 里做过了，
            # 三层重试）。
            next_step = "generate"

        return {
            "tool_results": records,
            "tool_messages": [assistant_msg, *tool_messages],
            "steps": state.steps + 1,
            "tokens_used": tokens,
            "next_step": next_step,
            "refusal": refusal if refused_by_model else None,
            "progress_keys": [progress_signature(tool_results=records)],
            "trace": [
                TraceStep(
                    step=state.steps + 1,
                    node="tool",
                    elapsed_ms=(_now() - t0) * 1000,
                    note="、".join(f"{r.tool}:{r.status}" for r in records) or "无调用",
                    detail=base_detail,
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #

    def generate(self, state: AgentState) -> dict[str, Any]:
        """按注入的法条和工具结果写答案，然后**立刻做引用校验**。

        引用校验放在这里而不是单独一个节点，是因为校验结果要用来**改答案**
        （删掉编造的编号），这是同一件事的两半。分开成两个节点会让中间的
        状态多一个"半成品答案"，没有价值。
        """
        t0 = _now()
        inv = self.deps.checker(state)
        violations = inv.generate_needs_context(state)
        if violations:
            # dev 模式走不到这里（上面抛异常了）。prod 模式降级成拒答。
            return {
                "steps": state.steps + 1,
                "next_step": "refuse",
                "refusal": prompts.build_refusal(
                    category="no_evidence",
                    reason="没有检索到可依据的法条，无法给出有出处的回答",
                    suggestion=(
                        "换一种问法再试一次（例如用更标准的法律术语：把「老板不给钱」"
                        "换成「拖欠劳动报酬」）；若确认不属于劳动法范畴，可咨询对应领域的专业人士"
                    ),
                ),
                "invariant_violations": violations,
                "progress_keys": ["generate:no-context"],
                "trace": [
                    TraceStep(
                        step=state.steps + 1,
                        node="generate",
                        elapsed_ms=(_now() - t0) * 1000,
                        note="无上下文 → 拒绝生成",
                    )
                ],
            }

        llm = self.deps.llm
        if llm is None:
            return {
                "steps": state.steps + 1,
                "errors": ["LLM 未装配，无法生成答案"],
                "next_step": "refuse",
                "refusal": prompts.build_refusal(
                    category="internal_error",
                    reason="服务未正确装配大模型客户端",
                    suggestion="请联系维护者",
                ),
                "progress_keys": ["generate:no-llm"],
                "trace": [
                    TraceStep(step=state.steps + 1, node="generate", note="LLM 缺失")
                ],
            }

        prompt = prompts.build_answer_prompt(
            state.question,
            state.injected,
            state.tool_results,
        )
        try:
            resp = llm.chat(
                [
                    {"role": "system", "content": prompts.ANSWER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=self.deps.settings.answer_max_tokens,
            )
            draft = resp.text
            tokens = state.tokens_used + (resp.usage.total_tokens or 0)
            errs: list[str] = []
        except Exception as e:  # noqa: BLE001
            draft = ""
            tokens = state.tokens_used
            errs = [f"生成失败：{type(e).__name__}: {e}"]

        report = verify_citations(draft, state.injected)
        cleaned = strip_invalid(draft, report)
        keys = resolve_keys(report, state.injected)

        return {
            "draft": cleaned,
            "answer": cleaned,
            "citations": keys,
            "citation_report": report,
            "steps": state.steps + 1,
            "tokens_used": tokens,
            "errors": errs,
            # 编号超范围一律记一笔（INV-02 在 dev 下会直接抛）。
            # 生产下不抛——**编造的编号已经被 strip_invalid 删掉了**，
            # 用户不会看到假引用，所以是"记下来 + 继续"。
            "invariant_violations": inv.require(
                not report.invalid,
                "INV-02",
                f"答案引用了不存在的编号 {report.invalid}，已删除",
                invalid=report.invalid,
                valid_range=sorted(a.index for a in state.injected),
            ),
            "next_step": "finalize",
            # ⚠️ **不要无条件写 `answered`**：这次生成可能是"撞闸门之后强行收尾"
            # 派发过来的，那 `finish_reason` 已经被闸门写成了 `guard_repeat` 之类。
            # 覆盖掉它，报表里就再也看不出"这条轨迹是被闸门掐断的"——
            # 那正是防死循环效果统计最需要的信息。所以：闸门的原因**优先保留**。
            "finish_reason": (
                state.finish_reason
                if (state.finish_reason or "").startswith("guard_")
                else ("answered" if cleaned.strip() else "refused")
            ),
            "progress_keys": [progress_signature(produced_answer=True)],
            "trace": [
                TraceStep(
                    step=state.steps + 1,
                    node="generate",
                    elapsed_ms=(_now() - t0) * 1000,
                    note=f"引用 {len(report.valid)}/{len(state.injected)} 条"
                    + (f"，删掉编造编号 {report.invalid}" if report.invalid else ""),
                    detail={
                        "cited": report.cited,
                        "invalid": report.invalid,
                        "precision": round(report.precision, 3),
                    },
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 拒答 / 闲聊
    # ------------------------------------------------------------------ #

    def refuse(self, state: AgentState) -> dict[str, Any]:
        """程序决定的拒答。**三种成因，三种措辞。**

        措辞必须分开：用户看到"这个问题超出我的范围"和"我没找到相关法条"
        是两种完全不同的体验，能采取的行动也不一样（前者去问别人，
        后者换个问法）。全用一句"抱歉我无法回答"是最偷懒也最没用的做法。
        """
        t0 = _now()
        if state.refusal:
            # 模型主动调了 refuse_to_answer，它自己给了理由和类别
            refusal = state.refusal
        elif state.route == "out_of_scope":
            refusal = prompts.build_refusal(
                category="out_of_scope",
                reason="这个问题不属于劳动法咨询范围",
                suggestion=(
                    "本助手只处理劳动法相关问题（劳动合同、解除、加班、社保、"
                    "试用期、仲裁时效等）。你这个问题建议咨询对应领域的专业人士。"
                ),
            )
        elif not state.injected:
            refusal = prompts.build_refusal(
                category="no_evidence",
                reason="没有检索到与你问题相关的法条，无法给出有依据的回答",
                suggestion=(
                    "换个说法再试一次，尽量用标准法律术语"
                    "（例如「拖欠劳动报酬」而不是「老板不给钱」）；"
                    "或者直接说明关键事实（工作了多久、月薪多少、怎么被辞退的），"
                    "信息越全，检索越准。"
                ),
            )
        else:
            refusal = prompts.build_refusal(
                category="insufficient_info",
                reason="现有信息不足以给出可靠结论",
                suggestion="补充关键事实（工作年限、月工资、解除原因、发生日期）后再问一次。",
            )
        return {
            "refusal": refusal,
            "answer": refusal.get("reason", ""),
            "steps": state.steps + 1,
            "finish_reason": "refused",
            "next_step": "finalize",
            "progress_keys": [progress_signature(produced_answer=True)],
            "trace": [
                TraceStep(
                    step=state.steps + 1,
                    node="refuse",
                    elapsed_ms=(_now() - t0) * 1000,
                    note=refusal.get("category", ""),
                )
            ],
        }

    def chitchat(self, state: AgentState) -> dict[str, Any]:
        """闲聊。**有模型就用模型回（自然），没模型就用固定文案（能用）。**"""
        t0 = _now()
        fallback = (
            "你好！我可以帮你查劳动法条文、算经济补偿金和加班费、看劳动合同是否合法。"
            "把你的情况说一下就行（比如「签了三年合同，试用期六个月合法吗」）。"
        )
        text = fallback
        tokens = state.tokens_used
        if self.deps.llm is not None:
            try:
                resp = self.deps.llm.chat(
                    [
                        {"role": "system", "content": prompts.CHITCHAT_SYSTEM},
                        {"role": "user", "content": state.question},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                text = resp.text.strip() or fallback
                tokens += resp.usage.total_tokens or 0
            except Exception:  # noqa: BLE001 — 闲聊回失败不配让用户等或报错
                text = fallback
        return {
            "answer": text,
            "draft": text,
            "steps": state.steps + 1,
            "tokens_used": tokens,
            "next_step": "finalize",
            "finish_reason": "answered",
            "progress_keys": [progress_signature(produced_answer=True)],
            "trace": [
                TraceStep(step=state.steps + 1, node="chitchat", elapsed_ms=(_now() - t0) * 1000)
            ],
        }

    # ------------------------------------------------------------------ #
    # 闸门 + 收尾
    # ------------------------------------------------------------------ #

    def guard(self, state: AgentState) -> dict[str, Any]:
        """四道防死循环的闸门。**所有工作节点都从这里过一道。**

        判定逻辑全在 `guard.check()` 里，这个节点只负责"把判定结果落到状态上"。

        ## ⚠️ 撞闸门之后不是立刻结束，而是"再生成一次然后结束"

        这一点很容易做错，而且两种做法都能跑通，只有一种体验是对的：

        - **错的做法**：撞闸门 → 直接去 `finalize`。结果：工具已经把它算出来了
          （"3 年 = 3 个月工资 = 30000 元"），检索到的法条也在手里，
          **但用户收到一个空白回复**。钱花了、料有了、东西没给。
        - **对的做法**（这里的做法）：撞闸门 → **不再允许新的工具调用**，
          但允许**用现有信息再生成一次答案**，然后无论成败都收尾。

        所以闸门收紧的是"还能不能继续调工具"，不是"要不要给用户答案"。
        对应地，`forced_final` 这个标志保证**强行收尾只发生一次**——
        否则 `guard → generate → guard` 会无限循环，**防死循环的闸门自己死循环**。

        **唯一的例外是 token 预算**：撞了预算还去生成，等于"钱花超了再多花一次"。
        所以那一种直接去 `finalize`，认了。
        """
        decision = check(state, self.deps.settings)
        if decision.ok:
            return {
                "trace": [
                    TraceStep(
                        step=state.steps,
                        node="guard",
                        note=(
                            f"通过（{state.steps}/{self.deps.settings.guard.max_steps} 步，"
                            f"{state.tokens_used} token）"
                        ),
                    )
                ]
            }

        reason_map = {
            "max_steps": "guard_steps",
            "repeat_call": "guard_repeat",
            "no_progress": "guard_no_progress",
            "token_budget": "guard_budget",
        }
        reason = reason_map.get(decision.limit or "", "error")

        # ⚠️ **同一个闸门在一条轨迹里只记一次。**
        #
        # 为什么需要这个去重：撞闸门之后会"强行收尾一次"，而收尾走的是
        #   guard（撞）→ generate（强行生成）→ guard（**又经过这里**）→ finalize
        # 第二次经过时 `state` 里的证据（那两次重复调用、那个步数）一点没变，
        # 所以 `check()` 会**原样再报一次同样的闸门**。
        #
        # 不去重的话有两处会坏：
        #   1. `guard_hits` 里同一条闸门出现两次 → 报表统计"被闸门掐断的轨迹数"
        #      会翻倍，而这个数字正是 F17 要交付的结论（"死循环率 X% → 0%"）。
        #   2. 那条 trace 的文案会变成"已停止工具调用，用现有信息强行收尾一次"，
        #      可这一次根本没再强行收尾（`forced_final` 已经是 True）——**日志撒谎**。
        #
        # 判据是"这一条闸门在这条轨迹里报过没有"，不是"是不是第二次经过闸门"：
        # 万一以后出现 ②报完又撞④的情况（换闸门了），那还是要照报的。
        already = any(h.limit == (decision.limit or "") for h in state.guard_hits)


        # 还能不能强行收尾一次？四个条件缺一不可，每个都有理由：
        #   `not state.answer`      —— 已经有答案了就不用再生成
        #   `not state.forced_final` —— 强行收尾只能一次（防自己死循环）
        #   `state.injected`        —— 没检索到东西就别生成了，那一定会编
        #   `route != out_of_scope` —— 越界问题生成什么？直接拒答
        #   加上 token 预算的例外。
        can_force = (
            not state.answer
            and not state.forced_final
            and bool(state.injected)
            and state.route != "out_of_scope"
            and decision.limit != "token_budget"
        )
        note = f"🚧 {decision.limit}：{decision.message}"
        if can_force:
            note += "。已停止工具调用，用现有信息强行收尾一次"
        elif already:
            note = f"🚧 {decision.limit}（本条轨迹里已记录过，不重复计数）"
        return {
            # 去重之后这里可能是空列表——空的增量写进状态里是安全的，
            # `Annotated[list, operator.add]` 加空列表等于没加。
            "guard_hits": [] if already else [decision.as_hit(state.steps)],
            "finish_reason": reason,
            "finish_detail": decision.message,
            # 粘住：一旦用过强行收尾，就一直为 True。
            # 写成 `can_force` 会让"第二次经过闸门"把它重置成 False，
            # 状态里就看不出**这条轨迹到底有没有走过强行收尾**——
            # 而这个信息要进报表（"防死循环救回了多少条作答"）。
            "forced_final": state.forced_final or can_force,
            "next_step": "generate" if can_force else "finalize",
            "trace": [
                TraceStep(
                    step=state.steps,
                    node="guard",
                    note=note,
                    detail={**decision.detail, "forced_final": can_force},
                )
            ],
        }

    def finalize(self, state: AgentState) -> dict[str, Any]:
        """收尾：跑最后几条断言、补齐 finish_reason。

        ⚠️ 收尾**不产生新内容**。撞闸门时已经拿到的答案照样给用户——
        半条答案也比"系统忙，请稍后重试"强。这一点想清楚很重要：
        闸门是"别再花钱了"，不是"把已有结果扔掉"。
        """
        t0 = _now()
        inv = self.deps.checker(state)
        violations: list[str] = []
        violations += inv.steps_within_budget(state, self.deps.settings.guard.max_steps)
        violations += inv.answer_nonempty(state)
        violations += inv.refusal_shape(state)
        violations += inv.citation_keys_injected(state)

        reason = state.finish_reason
        detail = state.finish_detail
        if reason is None:
            # 走到这里没定过 finish_reason，说明有节点"没说要收尾就来了"。
            # 不猜，记下来并当作 answered——但如果是空的，下面 answer_nonempty 会报。
            reason = "answered"
            detail = detail or "未显式设置 finish_reason，按 answered 处理"
            violations.append(
                "[INV-09] 收尾时 finish_reason 为空。有节点跳过了规则直接到 finalize。"
            )

        return {
            "finish_reason": reason,
            "finish_detail": detail,
            "invariant_violations": violations,
            "trace": [
                TraceStep(
                    step=state.steps,
                    node="finalize",
                    elapsed_ms=(_now() - t0) * 1000,
                    note=f"{reason}"
                    + (f"（{len(violations)} 条断言违规）" if violations else ""),
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 内部：给工具循环拼消息
    # ------------------------------------------------------------------ #

    def _agent_messages(self, state: AgentState) -> list[dict[str, Any]]:
        """工具循环用的消息列表。

        结构固定为：系统提示 → （可选）多轮历史 → 本轮问题 → 之前几轮的工具往返。

        ⚠️ **历史只带最近 6 条**。这不是偷懒，是 F20 的工作记忆：
        20 轮对话全塞进去，光历史就吃掉几千 token，而真正决定这次回答的
        是"这一轮的问题 + 这几轮的工具结果"。**更久远的信息该由槽位和摘要承载**
        （F19/F21），不该靠无限堆消息。
        """
        msgs: list[dict[str, Any]] = [{"role": "system", "content": prompts.AGENT_SYSTEM}]
        for m in state.history[-6:]:
            role = str(m.get("role", "user"))
            if role not in ("user", "assistant"):
                continue
            msgs.append({"role": role, "content": str(m.get("content", ""))})
        msgs.append({"role": "user", "content": state.question})
        msgs.extend(state.tool_messages)
        return msgs


__all__ = ["AgentDeps", "Nodes"]
