"""图与节点的端到端测试（F15/F16 的验收）。

## 这一层测的是什么，不是什么

**测**：链路的**形状**——四类输入走对分支、工具循环转得起来、消息带得全、
闸门拦得住、引用校验接得上、trace 完整。

**不测**：答案好不好。那需要真模型和真语料，属于评测层（`reports/` 那一套）。
用假模型测"答案质量"没有意义——你测的是**你自己的假答案**。

`ScriptedLLM` 的价值就在这里：它让"模型说了什么"变成一个**受控输入**，
于是"链路对这个输入做了什么"才成为一个可以被断言的问题。

## 一个反复出现的断言模式：断言"结构性质"而不是"具体文本"

比如 `test_tool_loop_carries_previous_round_messages` 断言的是
"第二轮的消息数比第一轮多"，而不是"第二条消息的内容是 xxx"。
**结构性质更稳**（改提示词不会让它红），而且**它才是真正会出 bug 的地方**——
内容错了人一眼能看出来，消息丢了是静默的。
"""

from __future__ import annotations

import pytest

from juris_agent.agent.graph import make_runner
from juris_agent.agent.state import AgentState
from juris_agent.config import Settings
from juris_agent.tools.builtin import build_registry
from juris_agent.tools.executor import ToolExecutor
from tests.fakes import FakeRetriever, ScriptedLLM, raw_tool_call, tool_call


def _settings(env: str = "prod") -> Settings:
    """测试默认用 **prod** 模式。

    理由：prod 模式下断言"记录但不抛"，测试能跑完整条链路并检查最终状态；
    dev 模式下第一个断言就会抛异常，只能测到断言本身。
    **dev 的行为由 `tests/test_invariants.py` 专门测**，这里测链路。
    """
    return Settings(env=env, deepseek_api_key=None)


def _runner(llm, retriever=None, *, env="prod", registry=None, executor=None):
    s = _settings(env)
    reg = registry if registry is not None else build_registry(retriever)
    ex = executor if executor is not None else ToolExecutor(reg, s)
    return make_runner(
        retriever=retriever,
        registry=reg,
        executor=ex,
        llm=llm,
        settings=s,
    )


# --------------------------------------------------------------------------- #
# ① 法条路：检索 → 生成（确定性管道，不用模型决定要不要查）
# --------------------------------------------------------------------------- #


def test_law_path_retrieves_then_generates() -> None:
    r = FakeRetriever()
    st = _runner(ScriptedLLM(["按第四十七条 [1]，每满一年一个月工资。"]), r).ask("被辞退有补偿吗？")
    assert st.route == "law"
    assert st.finish_reason == "answered"
    assert st.citations == ["lcl#47"]
    assert [c[0] for c in r.calls] == ["被辞退有补偿吗？"], "检索应该只发生一次"
    assert [t.node for t in st.trace] == [
        "entry", "retrieve", "guard", "generate", "guard", "finalize",
    ]


def test_law_path_stores_full_context_with_sequential_indexes() -> None:
    """编号必须**从 1 开始连续**——`verify.py` 的整套校验建立在这个前提上。"""
    st = _runner(ScriptedLLM(["看 [1] 和 [2]。"]), FakeRetriever()).ask("辞退有补偿吗")
    assert [a.index for a in st.injected] == [1, 2]
    assert len(st.injected) == 2


def test_retrieval_empty_refuses_without_calling_the_model() -> None:
    """**检索为空时一次模型调用都不该发生。**

    没依据时模型的默认行为是"用通用法律知识补一段听起来合理的话"——
    那正是本项目最不能接受的输出。所以这条路必须在**调用模型之前**就断掉。

    ⚠️ 这个断言的形状（`llm.msg_counts == []`）比断言"答案是拒答"更强：
    前者证明了"钱没白花"，后者只证明了"文案对"。
    """
    llm = ScriptedLLM(["不该被调用"])
    st = _runner(llm, FakeRetriever(hits=[])).ask("劳动合同法怎么规定的")
    assert st.finish_reason == "refused"
    assert st.refusal and st.refusal["category"] == "no_evidence"
    assert llm.msg_counts == [], "检索为空时不该调用模型"


# --------------------------------------------------------------------------- #
# ② 计算路：检索 → 工具循环 → 生成
# --------------------------------------------------------------------------- #

# ⚠️ 下面这些剧本为什么比"看起来该有的"多一条？
#
# 因为工具循环结束**要靠问模型一次**：模型返回"我没有工具要调了"，
# 我们才知道它说完了。那一轮的正文会被丢掉（`generate` 节点会用带编号的
# 上下文重新问一次，好让答案带上可校验的引用）。所以一条 N 轮工具调用的轨迹
# 需要 N 条"调工具"的剧本 + 1 条"我不调了" + 1 条"生成答案"。
#
# **这是真实的成本**：计算题比法条题多一次模型调用。换来的是答案里有编号。
# 想省这笔钱的做法是"把工具循环最后一轮的正文直接当答案"，代价是那段正文
# 是在**没有编号上下文**的提示词下写出来的，很可能不带引用——
# 那就等于放弃了引用校验。这里选择多花一次调用，理由写在 `nodes.py` 里。


def test_calc_path_runs_the_tool_loop() -> None:
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "calc_severance",
                                      {"cause": "economic_layoff",
                                       "months_of_service": 36,
                                       "avg_monthly_wage": 10000})]},
            {"text": "（工具循环收尾轮，正文会被丢掉）"},
            {"text": "按第四十七条 [1]，你应得 30000 元。"},
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("公司裁员，我干了3年月薪1万，能赔多少？")
    assert st.route == "calc"
    assert st.finish_reason == "answered"
    assert [t.tool for t in st.tool_results] == ["calc_severance"]
    assert st.tool_results[0].status == "ok"
    # ⚠️ 要 `float(...)` 转一下。工具出参走的是 `model_dump(mode="json")`，
    # 金额字段是 `Decimal`，序列化成 JSON 之后**是字符串**（"30000.00"）。
    # 直接拿去比大小会在 Python 3 里抛 TypeError（字符串和数字不能比），
    # 而如果字段恰好是数字，测试就静默通过了——所以这里显式转换、显式断言类型。
    total = st.tool_results[0].payload["total_payable"]
    assert float(total) > 0, f"应得金额应该是正数，实际 {total!r}"


def test_calc_path_mixes_search_then_calculate() -> None:
    """F16 的验收："先查法条再算钱"。

    这条测试走完整的两轮工具调用：第一轮 `search_law` 查条文，
    第二轮 `calc_severance` 算金额。**两轮都要在 trace 里看得见**
    （`trace` 里 `tool` 出现两次）。
    """
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "search_law", {"query": "经济补偿 计算标准"})]},
            {"tool_calls": [tool_call("c2", "calc_severance",
                                      {"cause": "economic_layoff",
                                       "months_of_service": 36,
                                       "avg_monthly_wage": 10000})]},
            {"text": "（收尾轮）"},
            {"text": "依据第四十七条 [1] 计算，共 30000 元。"},
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("经济补偿能拿多少？我干了3年月薪1万")
    assert [t.tool for t in st.tool_results] == ["search_law", "calc_severance"]
    # 工具节点走了 **3** 次而不是 2 次：两次真实调用 + 最后那次"我不调了"。
    # 这个数字把上面那段注释里的成本讲清楚了——**3 次模型调用才算完一道计算题**。
    assert [t.node for t in st.trace].count("tool") == 3


def test_tool_loop_carries_previous_round_messages() -> None:
    """**第二轮模型调用必须看到第一轮的往返。**

    看不到的话模型会以为"还没查过"，于是重复调一遍同一个工具——
    这是"Agent 原地打转"的一个真实成因，而且是**因为消息没带全**导致的，
    不是模型笨。断言消息条数递增就是在测这件事。
    """
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "search_law", {"query": "q1"})]},
            {"tool_calls": [tool_call("c2", "search_law", {"query": "q2"})]},
            {"text": "（收尾轮）"},
            {"text": "好 [1]"},
        ]
    )
    _runner(llm, FakeRetriever()).ask("经济补偿怎么算，我干了一年半")
    assert len(llm.msg_counts) >= 2
    assert llm.msg_counts[1] > llm.msg_counts[0], (
        f"第二轮消息数没有增加（{llm.msg_counts}），说明工具往返的消息丢了"
    )


def test_tools_are_offered_to_the_model() -> None:
    llm = ScriptedLLM([{"text": "（收尾轮）"}, {"text": "好 [1]"}])
    _runner(llm, FakeRetriever()).ask("加班费怎么算")
    assert llm.tool_flags and llm.tool_flags[0] is True


def test_bad_json_arguments_become_a_retryable_tool_error() -> None:
    """模型给出非法 JSON 参数时，**不崩、不猜**，而是把错误喂回去让它重试。

    猜它想传什么是最坏的做法：猜错了会执行一个**用户没要求过的计算**，
    而结果是看起来正常的金额——这种错误没法被发现。
    """
    llm = ScriptedLLM(
        [
            {"tool_calls": [raw_tool_call("c1", "calc_severance", "{cause: 不是JSON")]},
            {"text": "信息不足 [1]。"},
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("裁员补偿怎么算，我干了三年月薪一万")
    assert st.tool_results[0].status == "failed"
    assert "JSON" in (st.tool_results[0].error or {}).get("message", "")
    assert st.finish_reason == "answered", "不该因为一次参数错误就整条失败"
    # 那条喂回去的 tool 消息里带着重试提示
    tool_msgs = [m for m in st.tool_messages if m.get("role") == "tool"]
    assert tool_msgs and "retry" in tool_msgs[0]["content"]
    # 参数全部失败 → 不再给模型一轮，直接生成
    assert [t.node for t in st.trace].count("tool") == 1


def test_model_can_refuse_via_tool() -> None:
    """模型主动调用 `refuse_to_answer` → 走拒答收尾，**不算失败**。

    ⚠️ 这条测的是**机制**（模型拒答时链路怎么走），不是"这个问题该不该拒"。
    所以问题本身选了一句能路由到计算路的句子——只有计算路才有工具循环，
    而模型主动拒答这个机制只存在于工具循环里。
    """
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "refuse_to_answer",
                                      {"reason": "缺少关键事实，无法计算",
                                       "category": "insufficient_info",
                                       "suggestion": "请补充工作年限与月工资"})]},
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("竞业限制能赔多少钱")
    assert st.finish_reason == "refused"
    assert st.refusal and st.refusal["category"] == "insufficient_info"


def test_all_tools_failing_goes_straight_to_generate() -> None:
    """工具全挂 → 不再给模型一轮。

    再给一轮它大概率会用同样的参数再试一次（那不叫重试，叫烧钱——
    真正的重试已经在 `executor` 里做过三层了）。
    所以剧本里只有 **1** 个工具轮 + 1 个生成轮。
    """
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "calc_severance", {"cause": "不存在的枚举"})]},
            {"text": "信息不足，无法计算 [1]。"},
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("裁员补偿怎么算，干了三年月薪一万")
    assert st.tool_results[0].status == "failed"
    assert st.finish_reason == "answered"
    assert [t.node for t in st.trace].count("tool") == 1, "工具全挂之后不该再要一轮"


# --------------------------------------------------------------------------- #
# ③ 越界 / ④ 闲聊
# --------------------------------------------------------------------------- #


def test_out_of_scope_refuses_without_retrieval_or_model() -> None:
    llm = ScriptedLLM(["不该被调用"])
    r = FakeRetriever()
    st = _runner(llm, r).ask("离婚时财产怎么分")
    assert st.route == "out_of_scope"
    assert st.finish_reason == "refused"
    assert st.refusal["category"] == "out_of_scope"
    assert r.calls == [] and llm.msg_counts == []


def test_chitchat_answers_directly() -> None:
    st = _runner(ScriptedLLM(["你好！我可以帮你查劳动法条文。"]), FakeRetriever()).ask("你好")
    assert st.route == "chitchat"
    assert st.finish_reason == "answered"
    assert "劳动法" in st.answer


def test_chitchat_falls_back_without_llm() -> None:
    """没有模型也要能回——**闲聊不该因为后端挂了就没有响应**。"""
    st = _runner(None, FakeRetriever()).ask("你好")
    assert st.route == "chitchat"
    assert st.answer.strip()


# --------------------------------------------------------------------------- #
# 防死循环在图上的表现
# --------------------------------------------------------------------------- #


def test_repeat_call_stops_the_loop_but_keeps_the_answer() -> None:
    """撞闸门之后**答案还在**——这是本图最容易被做错的地方。

    "错的做法"是撞闸门直接结束：工具已经把钱算出来了、法条也在手里，
    但用户收到一个空白回复。钱花了、料有了、东西没给。
    所以正确行为是：停止新的工具调用，但允许**用现有信息再生成一次**。
    """
    args = {"cause": "economic_layoff", "months_of_service": 36, "avg_monthly_wage": 10000}
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "calc_severance", args)]},
            {"tool_calls": [tool_call("c2", "calc_severance", args)]},  # 一模一样的第二次
            {"text": "按第四十七条 [1]，你能拿 30000 元。"},               # 强行收尾那一次
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("公司裁员，干了三年月薪一万能赔多少")
    assert st.finish_reason == "guard_repeat", "结束原因要记下是被闸门掐的"
    assert st.answer.strip(), "但仍要给出答案"
    assert st.forced_final is True
    assert [h.limit for h in st.guard_hits] == ["repeat_call"]


def test_guard_forces_only_one_final_generation() -> None:
    """`forced_final` 保证强行收尾**只发生一次**——否则闸门自己会死循环。

    这里断言的是 LLM 被调用的次数是**有限的**（而不是跑不完被测试超时挂掉）。
    """
    args = {"cause": "economic_layoff", "months_of_service": 36, "avg_monthly_wage": 10000}
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "calc_severance", args)]},
            {"tool_calls": [tool_call("c2", "calc_severance", args)]},
            {"text": "答案 [1]"},
        ]
    )
    # ⚠️ 问题里**必须有计算信号**（这里靠「能赔多少」），否则会路由到 law 路，
    # 而 law 路没有工具循环——那么"强行收尾"这条路径根本走不到，
    # 测试会因为"路由错了"而红，跟它想测的东西完全无关。
    st = _runner(llm, FakeRetriever()).ask("裁员补偿能赔多少，干了三年月薪一万")
    assert len(llm.msg_counts) == 3, f"应该正好三次模型调用，实际 {len(llm.msg_counts)}"
    assert st.finish_reason == "guard_repeat"


def test_max_steps_stops_a_model_that_keeps_calling_tools() -> None:
    """模型一直换着参数调工具 → 步数上限兜底。

    注意这里每轮参数都不同，所以**重复调用闸门不会触发**——
    这正好说明为什么需要①：②接不住的形态必须有人接。
    """
    s = _settings()
    s.guard.max_steps = 4
    s.guard.no_progress_limit = 99  # 只留步数闸门，隔离出①的行为
    # **用 `tail=` 而不是把剧本写死。** 这三条测试关心的是"最后停没停"，
    # 不关心"第几轮停的"——把轮次写死会让测试在闸门参数微调时无谓地变红，
    # 而且写死的那个数字本身就是从实现里抄来的（测了个寂寞）。
    script = [
        {"tool_calls": [tool_call(f"c{i}", "search_law", {"query": f"q{i}"})]}
        for i in range(20)
    ]
    llm = ScriptedLLM(script, tail={"text": "强行收尾 [1]"})
    reg = build_registry(FakeRetriever())
    st = make_runner(
        retriever=FakeRetriever(), registry=reg, executor=ToolExecutor(reg, s), llm=llm, settings=s
    ).ask("经济补偿怎么算")
    assert st.finish_reason == "guard_steps"
    # 上限是 `4 + 1`：撞上第 4 步的闸门之后，图**故意**再走一步"强行收尾"
    # （用已经拿到的东西生成答案，而不是给用户一个空白回复）。
    # 所以"停住了"的判据是 `max_steps + 1` 而不是 `max_steps`。
    # 断言成 `<= 4` 会把这条正确的设计判成失败。
    assert st.steps <= s.guard.max_steps + 1, f"步数 {st.steps} 超过了上限+强行收尾"


def test_no_progress_stops_a_model_that_repeats_itself() -> None:
    """模型反复"想不起来自己查过" → 无进展闸门。"""
    s = _settings()
    s.guard.no_progress_limit = 2
    s.guard.max_steps = 99
    # 每轮参数都不同（所以②不触发），但检索器永远返回同样的结果
    script = [
        {"tool_calls": [tool_call(f"c{i}", "search_law", {"query": f"第{i}次问"})]}
        for i in range(10)
    ]
    llm = ScriptedLLM(script, tail={"text": "强行收尾 [1]"})
    reg = build_registry(FakeRetriever())
    st = make_runner(
        retriever=FakeRetriever(), registry=reg, executor=ToolExecutor(reg, s), llm=llm, settings=s
    ).ask("经济补偿怎么算")
    assert st.finish_reason == "guard_no_progress"


def test_budget_guard_does_not_spend_more() -> None:
    """撞 token 预算就**不再生成**——钱超了还去生成等于"再多花一次"。"""
    s = _settings()
    s.guard.token_budget = 5  # 第一次模型调用就会超
    s.guard.max_steps = 99
    llm = ScriptedLLM(
        [
            {"tool_calls": [tool_call("c1", "search_law", {"query": "q"})], "tokens": 100},
        ],
        tail={"text": "不该被调用"},
    )
    reg = build_registry(FakeRetriever())
    st = make_runner(
        retriever=FakeRetriever(), registry=reg, executor=ToolExecutor(reg, s), llm=llm, settings=s
    ).ask("经济补偿怎么算")
    assert st.finish_reason == "guard_budget"
    assert len(llm.msg_counts) == 1, "超预算之后不该再调模型"


# --------------------------------------------------------------------------- #
# 引用校验接在链路上
# --------------------------------------------------------------------------- #


def test_fabricated_citation_is_stripped_and_recorded() -> None:
    """模型编了 `[9]`：**用户看不到它，但报表里看得到**。

    这两件事必须同时成立——只删不记，你永远不知道模型在编；
    只记不删，用户会点到一个空的出处。
    """
    st = _runner(ScriptedLLM(["依据 [1] 和 [9]，你可以要求赔偿。"]), FakeRetriever()).ask(
        "被辞退有补偿吗"
    )
    assert "[9]" not in st.answer
    assert "[1]" in st.answer
    assert st.citation_report.invalid == [9]
    assert any("INV-02" in v for v in st.invariant_violations)
    assert st.citations == ["lcl#47"], "无效引用不该进 citations"


def test_dev_mode_raises_on_fabricated_citation() -> None:
    """dev 模式下同一件事**直接炸**——开发期要它立刻暴露。

    这是"同一套断言、两种失败策略"的完整证据：
    同一个输入，prod 记一笔继续，dev 抛异常。
    """
    from juris_agent.agent.invariants import InvariantError

    with pytest.raises(InvariantError) as ei:
        _runner(ScriptedLLM(["依据 [9] 赔偿。"]), FakeRetriever(), env="dev").ask("被辞退有补偿吗")
    assert ei.value.code == "INV-02"


# --------------------------------------------------------------------------- #
# 流式（F29 的地基）
# --------------------------------------------------------------------------- #


def test_stream_yields_per_node_updates() -> None:
    """`stream()` 按节点产出**增量**。SSE 层靠这个把每一步推给前端。"""
    r = _runner(ScriptedLLM(["依据 [1]。"]), FakeRetriever())
    chunks = list(r.stream("经济补偿怎么算"))
    nodes = [n for c in chunks for n in c]
    assert "entry" in nodes and "retrieve" in nodes and "finalize" in nodes


# --------------------------------------------------------------------------- #
# 没有检索器时的行为（"只想试试图能不能跑"）
# --------------------------------------------------------------------------- #


def test_missing_retriever_refuses_cleanly() -> None:
    """检索器没装配 → 明确拒答并写错误，**不是抛异常带走整个请求**。"""
    st = _runner(None, None).ask("经济补偿怎么算")
    assert st.finish_reason == "refused"
    assert any("检索器" in e for e in st.errors)


def test_missing_llm_refuses_cleanly() -> None:
    st = _runner(None, FakeRetriever()).ask("经济补偿怎么算")
    assert st.finish_reason == "refused"
    assert st.refusal["category"] == "internal_error"


# --------------------------------------------------------------------------- #
# 状态对象的自检
# --------------------------------------------------------------------------- #


def test_state_seen_summary_shape() -> None:
    st = _runner(ScriptedLLM(["依据 [1]。"]), FakeRetriever()).ask("被辞退有补偿吗")
    assert isinstance(st, AgentState)
    assert st.seen_state().startswith("route=law")
