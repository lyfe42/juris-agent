"""记忆层接进主链路的验收（F19–F22 的"最后一根线"）。

## 这一层测的是什么，为什么它必须单独存在

`tests/test_memory.py` 测的是**记忆层自己**（槽位怎么抽、窗口怎么切、
预算怎么分）。那些全绿，**和"记忆层有没有真的跑在链路上"是两件事**。

这次的真题就出在这里：四层记忆写完了、消融报告也出了
（`reports/memory_ablation.md`），但图里**没有任何节点调它**。
`fill_args()` 是死代码。全套 619 条测试照样绿——**因为没有一条测试
问过"这个功能接上了吗"**。

所以这个文件的每一条测试，问的都是同一个问题的不同侧面：

> **用户第 1 轮说的话，第 20 轮问的时候，到底走到模型眼前了没有？**

具体拆成三根线，缺一根就白做：

| 线 | 断了的症状 | 这里的测试 |
|---|---|---|
| 记忆块进提示词 | 模型又回头问一遍"你月薪多少" | `test_*_prompt_*` |
| 槽位补全工具参数 | 模型知道月薪，填参数时还是漏 | `test_*prefill*` |
| 缺的事实翻译成中文追问 | 答案里出现"请提供 months_of_service" | `test_*missing*` |
"""

from __future__ import annotations

import pytest

from juris_agent.agent.graph import AgentRunner, make_runner
from juris_agent.agent.nodes import _current_turn, _is_missing_fact
from juris_agent.agent.state import AgentState
from juris_agent.config import Settings
from juris_agent.memory.manager import MemoryManager, MemoryPolicy, manager_from_settings
from juris_agent.tools.builtin import build_registry
from juris_agent.tools.executor import ToolExecutor
from tests.fakes import FakeRetriever, ScriptedLLM, tool_call

# --------------------------------------------------------------------------- #
# 脚手架
# --------------------------------------------------------------------------- #


def _settings(**memory) -> Settings:
    """`env="prod"`：断言"记录但不抛"，测试能跑完整条链路（理由同 test_agent_graph）。

    `memory` 只在显式给的时候才传——**不能写成 `memory=memory or None`**，
    pydantic 会拿 `None` 去校验一个嵌套模型然后报"Input should be a valid
    dictionary"。不给就是 None 的话，测的其实是"默认配置对不对"，
    而这里想测的是"我指定的那一项生效了没有"。
    """
    kw = {"memory": memory} if memory else {}
    return Settings(env="prod", deepseek_api_key=None, **kw)


def _runner(llm, retriever=None, *, settings=None, memory=None) -> AgentRunner:
    s = settings or _settings()
    reg = build_registry(retriever)
    return make_runner(
        retriever=retriever,
        registry=reg,
        executor=ToolExecutor(reg, s),
        llm=llm,
        settings=s,
        memory=memory,
    )


def _long_history() -> list[dict[str, str]]:
    """一段 19 轮的对话：**关键事实只在第 1 轮出现，后面谁都不再提**。

    这正是记忆层存在的理由。写成 19 轮而不是 2 轮，是因为
    "隔了多远"才是真的在被测的东西——隔 1 轮的话，随便什么策略都记得住。

    第 1 轮里塞了**算经济补偿要的全部三项**（月薪 / 工作年限 / 地点），
    这样后面每条测试都能靠"第 1 轮说过、第 20 轮还要用"这一个形状
    去测不同的链路，不用每条约会换一套历史。
    """
    hist = [{"role": "user", "content": "我在西安，月薪12000，干了两年，公司说要裁员"}]
    for i in range(2, 20):
        hist.append({"role": "assistant", "content": f"（第{i - 1}轮回答）已按劳动法相关条款分析。"})
        hist.append({"role": "user", "content": f"那我第{i}个问题：这种情况还有什么要注意的"})
    return hist


def _answer_prompt(llm: ScriptedLLM) -> str:
    """把**发给生成节点**的那条 user 消息取出来。

    生成是最后一次调用（前面还可能有工具循环），所以取 `messages[-1]` 里的
    最后一条 user 消息。断言"提示词里有没有那段记忆"必须看这个，
    不能看 `state.memory_block`——**那个字段有内容、但没送进去，是最典型的
    半数交付**（组装好了忘了用）。要是哪天有人删掉 `generate` 里那个
    `history_block=state.memory_block`，只有这一条测试会红。
    """
    assert llm.messages, "一次模型调用都没发生"
    last = llm.messages[-1]
    return "\n".join(str(m.get("content", "")) for m in last if m.get("role") == "user")


# --------------------------------------------------------------------------- #
# ① 记忆块：用户几十轮前说的事，得走到模型眼前
# --------------------------------------------------------------------------- #


def test_the_fact_from_turn_one_reaches_the_answer_prompt() -> None:
    """**这是本文件的核心一条。** 第 20 轮问"能赔多少"，提示词里必须出现第 1 轮的月薪。

    隔了 19 轮，早就出了 6 轮的滑动窗口。没有记忆层的话，
    模型看到的提示词里**既没有"12000"，也没有"西安"**——
    它只能反问用户，或者更糟：编一个数。
    """
    llm = ScriptedLLM(["按第四十七条 [1]，每满一年一个月工资。"])
    st = _runner(llm, FakeRetriever()).ask(
        "那到底能赔我多少钱", history=_long_history()
    )
    prompt = _answer_prompt(llm)
    assert "月工资：12000" in prompt, "第 1 轮的月薪没进提示词"
    assert "城市：西安" in prompt
    # 槽位表里带出处的那句话（"已核实的事实"是槽位层的表头）
    assert "已核实的事实" in prompt
    assert st.memory_policy == "layered"
    assert st.memory_tokens > 0


def test_the_prompt_gives_the_provenance_not_just_the_value() -> None:
    """光有"月工资 12000"不够，**还要能说出这是第几轮、原话是什么**。

    这是用户问"你怎么知道我工资多少"时唯一能拿出来的东西。
    要么把出处带上，要么在被问的时候答不出来——**事后是补不出来的**，
    因为那一轮的原文早就出窗口了。
    """
    llm = ScriptedLLM(["按第四十七条 [1]。"])
    st = _runner(llm, FakeRetriever()).ask("能赔多少", history=_long_history())
    prov = st.slots_provenance
    assert "monthly_wage" in prov
    assert "第1轮" in prov["monthly_wage"]
    assert "12000" in prov["monthly_wage"]


def test_a_fresh_manager_and_a_reused_one_agree() -> None:
    """**每轮新建管理器 / 一个会话复用一个，结果必须一样。**

    这条是"调用方可以偷懒"的保证：`AgentRunner` 默认每轮现造一个，
    靠 `catch_up()` 从历史里补喂槽位。要是哪天补喂坏了，
    表现是"多轮对话里越聊越傻"——而每轮单独测都是好的，**这条才抓得住**。
    """
    hist = _long_history()
    # 每轮新建（`ask` 不传 memory）
    llm_a = ScriptedLLM(["答 [1]。"])
    st_a = _runner(llm_a, FakeRetriever()).ask("能赔多少", history=hist)
    # 跨轮复用同一个
    mgr = manager_from_settings(_settings())
    llm_b = ScriptedLLM(["答 [1]。"])
    st_b = _runner(llm_b, FakeRetriever()).ask("能赔多少", history=hist, memory=mgr)
    assert st_a.memory_block == st_b.memory_block
    assert st_a.slots_provenance == st_b.slots_provenance


def test_the_memory_layer_can_be_turned_off() -> None:
    """开关关掉之后**真的什么都没做**——否则那不算开关，算装饰。

    留这个开关的理由：要证明"记忆层有用"，必须能把它整个摘掉跑一遍。
    顺带它也是故障时的应急闸（记忆层算出乱七八糟的东西时，
    先关掉它恢复服务，再慢慢查）。
    """
    s = _settings(enabled=False)
    llm = ScriptedLLM(["按第四十七条 [1]。"])
    # ⚠️ 问句**必须走法条路**（`被辞退有补偿吗` 命中 law）。
    # 写成"能赔多少"会路由到计算路，先走工具节点把剧本第一条吃掉，
    # 生成节点就没台词了 → 答案是空的 → `refused`。
    # 那和"关掉记忆"没关系，是测试自己写错了路由。
    st = _runner(llm, FakeRetriever(), settings=s).ask(
        "被辞退有补偿吗", history=_long_history()
    )
    assert st.memory_block == ""
    assert st.memory_tokens == 0
    # ⚠️ 判据用「已核实的事实」这个**槽位层的表头**，不能用"月工资"这种词：
    # 第四十七条的正文里就有"一个月工资"，拿它当判据永远为真——
    # 那种测试是**永远会过**的假测试，比没有还坏。
    assert "已核实的事实" not in _answer_prompt(llm)
    # 但链路本身必须照常跑通——关掉记忆不该让服务挂掉
    assert st.finish_reason == "answered"


def test_the_trace_shows_the_memory_step_with_its_layer_breakdown() -> None:
    """trace 里要能看到"每一层放了几条"。**报表和排障都靠它。**"""
    llm = ScriptedLLM(["答 [1]。"])
    st = _runner(llm, FakeRetriever()).ask("能赔多少", history=_long_history())
    step = next(t for t in st.trace if t.node == "remember")
    assert "第 20 轮" in step.note
    assert {lay["name"] for lay in st.memory_layers} >= {"slots", "recent"}
    assert step.detail["policy"] == "layered"


# --------------------------------------------------------------------------- #
# ② 槽位预填工具参数：模型知道 ≠ 模型会填
# --------------------------------------------------------------------------- #


def _severance_call(args: dict) -> dict:
    return {"tool_calls": [tool_call("c1", "calc_severance", args)]}


def test_prefill_saves_a_tool_call_whose_args_the_model_dropped() -> None:
    """**模型只填了一半参数，槽位把它补全，工具照样算得出来。**

    剧本刻意只给 `cause`——真实模型就是这样：看得到月薪，
    但填参数时只填它觉得"这一轮在问的那个"。结果是工具报缺字段、
    转一圈回来再问一遍用户（用户会崩溃："我第 1 轮就说了啊"）。
    """
    hist = _long_history()
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff"}),
            {"text": ""},  # 工具循环收尾：我不调了
            "按第四十七条 [1]，赔 3 个月。",
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("那到底能赔我多少钱", history=hist)

    ok = [r for r in st.tool_results if r.tool == "calc_severance" and r.status == "ok"]
    assert ok, f"预填没生效，工具失败了：{[r.error for r in st.tool_results]}"
    assert ok[0].args["avg_monthly_wage"] == 12000, "月薪应该由槽位补上"
    assert ok[0].args["months_of_service"], "工作年限应该由槽位补上"
    # 补参数这件事必须留痕——**静默补参数比不补更危险**：
    # 出了问题你会以为是模型填的，去查模型的提示词，永远查不到。
    step = next(t for t in st.trace if t.node == "tool" and t.detail.get("prefilled"))
    assert "avg_monthly_wage" in step.detail["prefilled"]


def test_prefill_can_be_switched_off_and_then_the_tool_fails() -> None:
    """关掉开关 → 工具报缺参数。**这是上一条测试的对照组。**

    没有这条，"预填有用"就只是一句话：谁知道是预填生效了，
    还是模型本来就填得全？**对照组才是证据。**
    """
    s = _settings(prefill_tool_args=False)
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff"}),
            {"text": ""},
            "按第四十七条 [1]。",
        ]
    )
    st = _runner(llm, FakeRetriever(), settings=s).ask(
        "那到底能赔我多少钱", history=_long_history()
    )
    rec = next(r for r in st.tool_results if r.tool == "calc_severance")
    assert rec.status == "failed"
    assert rec.error and rec.error["error_type"] == "ToolValidationError"


def test_prefill_never_overwrites_what_the_model_explicitly_filled() -> None:
    """模型明确填了、但是和用户原话对不上时：**按模型的算，但记一笔。**

    ⚠️ 这条是刻意的设计，不是没做完。为什么不当场纠正：

    槽位记的是"用户说过什么"，不是"这次该算什么"。
    用户第 1 轮说"月薪 12000"、第 9 轮说"里面含 2000 餐补"——
    该不该把餐补算进补偿基数，**是个法律判断，得模型来做**。
    程序在这里强行覆盖，就把一个法律判断降级成了一个字符串比较。
    所以选的是"缺的补上、有分歧的留痕"。

    ⚠️ 如果哪天有人把这里改成"直接覆盖"，**用户改了口的场景会静默算错**，
    而且错得让人无从查起（参数看着就是用户说的数）。
    """
    hist = _long_history()
    llm = ScriptedLLM(
        [
            _severance_call(
                {"cause": "economic_layoff", "avg_monthly_wage": 9000}
            ),
            {"text": ""},
            "按第四十七条 [1]。",
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("那到底能赔我多少钱", history=hist)
    rec = next(r for r in st.tool_results if r.tool == "calc_severance")
    assert rec.args["avg_monthly_wage"] == 9000, "不该覆盖模型明确填的值"
    # 但分歧必须留下一句话，且要带上"用户原话是什么、第几轮说的"
    conflict = next(e for e in st.errors if "avg_monthly_wage" in e)
    assert "第 1 轮" in conflict and "12000" in conflict


def test_prefill_happens_before_the_idempotency_key() -> None:
    """补参数必须发生在算幂等键**之前**，否则缓存永远命中不了。

    幂等键是 `hash(工具名 + 参数)`。模型漏填的那次和填全的那次，
    补完之后参数是一样的，**应该命中同一份缓存**；如果在补之前算键，
    两次的键不同 → 同一个计算做两遍，幂等缓存白装。

    这种顺序错误**不会抛异常**，只是成本悄悄翻倍。
    """
    hist = _long_history()
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff"}),
            _severance_call(
                {"cause": "economic_layoff", "avg_monthly_wage": 12000, "months_of_service": 24}
            ),
            {"text": ""},
            "按第四十七条 [1]。",
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("那到底能赔我多少钱", history=hist)
    calls = [r for r in st.tool_results if r.tool == "calc_severance"]
    assert len(calls) == 2
    assert calls[0].args == calls[1].args, "两次调用补完之后参数应该一致"
    assert calls[1].idempotency_hit, "第二次应该命中幂等缓存，不该重算"


# --------------------------------------------------------------------------- #
# ③ 缺的事实：翻译成中文去问用户，而不是让模型猜
# --------------------------------------------------------------------------- #


def test_a_missing_required_field_becomes_a_chinese_question() -> None:
    """工具报缺字段时，**用户看到的必须是中文标签，不是 `months_of_service`。**

    生产里最常见的路径：用户只说了"公司要裁员，能赔多少"——
    月薪和工作年限都没说。这时候正确答案是**问**，不是算。
    而"问什么"必须由槽位表来定：它知道哪些字段是"没有就算不出来"的。
    """
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff"}),
            {"text": ""},
            "还缺几项信息 [1]。",
        ]
    )
    # 历史很短，什么事实都没有
    st = _runner(llm, FakeRetriever()).ask("公司裁员能赔多少")
    assert "月工资" in st.missing_slots
    assert "工作年限（月）" in st.missing_slots
    assert all("_" not in s for s in st.missing_slots), "标签里不能出现字段名"
    assert "还缺哪些事实" in _answer_prompt(llm)


def test_only_the_fields_that_actually_went_missing_get_asked() -> None:
    """**只问真正缺的那一项，不要连模型已经处理好的也一起问。**

    模型填了 `cause`（解除情形）、漏了 `months_of_service`（工作年限）时，
    如果按"槽位表里缺哪些必填项"去问，会问出
    「请补充：解除情形、工作年限」——**第一个用户根本答不上来**，
    那不是他能提供的事实，是法律定性；而且模型明明已经处理好了。

    这一条正是把"槽位表缺什么"和"这次调用缺什么"分开的理由。
    """
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff"}),
            {"text": ""},
            "还缺几项信息 [1]。",
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("公司裁员能赔多少")
    assert "工作年限（月）" in st.missing_slots
    assert "解除情形" not in st.missing_slots, "模型已经填了 cause，不该再问用户"


def test_a_type_error_is_not_reported_as_a_missing_fact() -> None:
    """模型自己填错（类型不符）**不该去问用户**——那是模型的错。

    只看"校验失败就问用户"的话，模型把 `cause` 填成 `"经济裁员 "`
    （多个空格、还不是枚举值）也会变成"请问您是哪种解除情形"。
    用户莫名其妙，而且这条追问掩盖了真正的 bug。
    """
    # ⚠️ 关键在"**只**有类型错"：钱和年限都照常填（预填会补），
    # 唯独 cause 给一个不在枚举里的值。只要有一条 missing，这条测试就变了味——
    # 那时候报缺是对的，测的就是别的东西了。
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "查无此情形"}),
            {"text": ""},
            "依据不足 [1]。",
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("那到底能赔我多少钱", history=_long_history())
    rec = next(r for r in st.tool_results if r.tool == "calc_severance")
    assert rec.status == "failed"
    assert [f["type"] for f in rec.error["field_errors"]] == ["enum"], (
        f"这条测试的前提是「只有枚举错」，实际报的是 {rec.error['field_errors']}"
    )
    assert st.missing_slots == [], "类型/枚举错误不该变成对用户的追问"


def test_the_crash_of_the_tool_is_not_blamed_on_the_user() -> None:
    """`_is_missing_fact` 的直接单测：把三种失败摆在一起对照。

    三种失败走同一个函数，但只有第一种该问用户。
    把它们放在一条测试里，是因为**它们的区别只在一个字段上**
    （`error_type` / `direction` / 字段级 `type`），
    分开写三处的话，将来改判据很容易只改对两个。
    """
    assert _is_missing_fact(
        {
            "error_type": "ToolValidationError",
            "direction": "input",
            "field_errors": [{"loc": "months_of_service", "type": "missing"}],
        }
    )
    # 类型不符 → 模型的错
    assert not _is_missing_fact(
        {
            "error_type": "ToolValidationError",
            "direction": "input",
            "field_errors": [{"loc": "cause", "type": "enum"}],
        }
    )
    # 出参不符 → 工具实现有 bug，该报警不该问用户
    assert not _is_missing_fact(
        {
            "error_type": "ToolValidationError",
            "direction": "output",
            "field_errors": [{"loc": "total", "type": "missing"}],
        }
    )
    # 超时 / 熔断 / 没错误 → 都不是"用户没说"
    assert not _is_missing_fact({"error_type": "ToolTimeoutError"})
    assert not _is_missing_fact({"error_type": "CircuitOpenError"})
    assert not _is_missing_fact(None)


# --------------------------------------------------------------------------- #
# ④ 工具循环的消息：历史由记忆层统一决定，不能有两个地方各切一刀
# --------------------------------------------------------------------------- #


def test_the_tool_loop_gets_the_memory_block_instead_of_a_second_window() -> None:
    """记忆块要进**工具循环**的系统消息——不然模型选工具时看不到历史事实。

    ⚠️ 同时要盯住一个更隐蔽的点：**历史不能有两份窗口**。
    以前这里写死 `history[-6:]`，那是记忆层还不存在时的临时替身。
    两处都留着就会漂移：改了一处忘了另一处，
    表现是"生成节点看到的上下文和工具节点看到的不一样"——不报错，
    只是让"模型为什么没算对"的排查找不到北。
    """
    hist = _long_history()
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff"}),
            {"text": ""},
            "按第四十七条 [1]。",
        ]
    )
    st = _runner(llm, FakeRetriever()).ask("那到底能赔我多少钱", history=hist)
    tool_msgs = llm.messages[0]  # 工具循环是第一次调用
    system = next(m for m in tool_msgs if m["role"] == "system")
    assert "月工资：12000" in system["content"], "记忆块没进工具循环的系统提示词"
    # 系统消息 + 本轮问题 + 工具往返；**不再额外带一遍 6 轮历史**
    assert sum(1 for m in tool_msgs if m["role"] == "user") == 1
    assert st.memory_block


def test_the_tool_loop_still_works_when_memory_is_off() -> None:
    """退回老做法：没记忆层时照样带最近 6 条历史。**降级路径也要能跑。**"""
    s = _settings(enabled=False)
    llm = ScriptedLLM(
        [
            _severance_call({"cause": "economic_layoff", "avg_monthly_wage": 12000,
                             "months_of_service": 24}),
            {"text": ""},
            "按第四十七条 [1]。",
        ]
    )
    _runner(llm, FakeRetriever(), settings=s).ask("能赔多少", history=_long_history())
    roles = [m["role"] for m in llm.messages[0]]
    assert roles.count("user") >= 2, "关掉记忆后应该退回 history[-6:] 的老做法"


# --------------------------------------------------------------------------- #
# ⑤ 纯函数：轮次口径
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "history,expected",
    [
        ([], 1),
        ([{"role": "user", "content": "a"}], 2),
        ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], 2),
        ([{"role": "user", "content": "a"}] * 19, 20),
        # 助手消息开头（历史被裁剪过）不该让它算成第 2 轮
        ([{"role": "assistant", "content": "b"}], 1),
    ],
)
def test_current_turn_counts_user_messages(history, expected) -> None:
    """**轮次口径必须是"一条用户消息 = 一轮"。**

    踩过的坑：`enumerate(history)` 那种"一条消息一轮"的算法，
    让"窗口 6 轮"实际只保住了 3 轮对话，而所有文档、所有报表写的都是 6 轮。
    **这种错不会报错**，只会让"第 20 轮丢了信息"的排查全对着错的地方找。
    """
    assert _current_turn(history) == expected


def test_the_manager_does_not_refeed_turns_it_has_already_seen() -> None:
    """`catch_up` 只能补没看过的。**重复喂会让槽位的冲突记录爆掉。**

    `Slots.put()` 拒绝"模糊值覆盖精确值"时会往 `history` 里写一条记录。
    要是每轮都把全历史重喂一遍，同一条"第 9 轮的『一万出头』被拒"
    会在第 10 到 20 轮里各记一次，日志被淹掉，
    而真正该看的那条（第 9 轮那次）混在里面找不出来。
    """
    mgr = manager_from_settings(_settings())
    hist = _long_history()
    mgr.remember("第一问", [], current_turn=1)
    assert mgr.observed_upto == 1
    # 把同样的历史喂第二遍，不应该再补喂
    assert mgr.catch_up(hist, current_turn=20) == 18  # 2..19 共 18 轮
    assert mgr.catch_up(hist, current_turn=20) == 0, "已经看过的轮次不该重喂"


def test_remember_never_feeds_the_current_turn_twice() -> None:
    """当前轮由 `remember()` 自己喂，`catch_up` 不能顺手也喂一遍。

    喂两遍的后果：`Slots.put()` 第二条规则是"同样精确就覆盖"，
    同一个值被写两次，`turn` 还是当轮，看起来没问题——
    但 `observe()` 返回的 `changed` 会重复，trace 里
    "第 20 轮更新了哪几个槽位"就成了假数据。
    """
    mgr = MemoryManager(policy=MemoryPolicy.LAYERED)
    hist = [{"role": "user", "content": "月薪 8000"}]
    mgr.remember("月薪 9000", hist, current_turn=2)
    assert mgr.observed_upto == 2
    assert mgr.slots.get("monthly_wage") == 9000, "当前轮的改口必须生效"


def test_ask_creates_a_separate_memory_per_conversation() -> None:
    """**两次不传 memory 的 ask，必须拿到两份独立的记忆。**

    这是防串数据的最后一道：进程里共用一个管理器的话，
    A 用户的月薪会出现在 B 用户的答案里。**串数据一旦发生，
    日志里完全看不出来**——B 的提示词里就是多了一行看起来很正常的槽位。
    """
    r = _runner(ScriptedLLM(["答 [1]。"]), FakeRetriever())
    a = r.ask("我在西安，月薪 12000，被裁了")
    b = r.ask("我在上海，月薪 9000，被裁了")
    assert a.memory is not b.memory
    assert a.slots_provenance.get("monthly_wage", "").startswith("12000")
    assert b.slots_provenance.get("monthly_wage", "").startswith("9000")


def test_the_state_carries_the_manager_so_the_caller_can_inspect_it() -> None:
    """管理器挂在状态上（不是挂在 deps 上），**调用方要能拿到它**。

    会话服务要靠它：一个 `thread_id` 对应一个管理器，
    存在会话表里、下次请求取出来接着用（省掉每轮重抽历史）。
    """
    st = _runner(ScriptedLLM(["答 [1]。"]), FakeRetriever()).ask("月薪 12000，被裁了")
    assert isinstance(st.memory, MemoryManager)
    assert st.memory.slots.get("monthly_wage") == 12000
    # 而且要能被序列化后又还原（将来存会话表 / LangGraph checkpointer 都要走这一步）
    dumped = AgentState.model_validate(
        {k: v for k, v in st.model_dump().items() if k != "memory"}
    )
    assert dumped.memory is None
