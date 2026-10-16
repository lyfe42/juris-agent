"""门面（`juris_agent.client`）与命令行（`juris_agent.cli`）的测试。

## 这一层测什么

**测**：门面**没有做多余的事**——它把 `AgentState` 翻成人话时
没丢字段、没改行为、没吞异常；装配失败时给的是**能被看懂的**错误；
成本那笔账的 `billable` 标志没被抹掉。

**不测**：命令行的输出格式好不好看（那是人看的东西，
断言"字符串里有这几个字"只会让改一句文案就红一片）；
也**不测**真的能不能回答问题（那要加载 1GB 模型 + 真花钱，
属于评测层）。

## 全部走注入，不碰模型、不碰网络

`build_runtime(settings, retriever=FakeRetriever(), llm=ScriptedLLM([...]))`
——注入这条路是 `api/runtime.py` 一开始就留好的（那边的 43 个契约测试
用的也是它）。**门面复用它，所以这里测的装配行为和服务那边是同一套。**
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from juris_agent.api.runtime import Runtime, build_runtime
from juris_agent.api.sessions import SessionStore
from juris_agent.client import Answer, Citation, JurisAgent, NotReady, _commit_turn
from juris_agent.cli import _answer_of, _print_answer, build_parser, main
from juris_agent.config import Settings
from juris_agent.llm import Usage
from tests.fakes import FakeRetriever, ScriptedLLM


def _settings(env: str = "prod") -> Settings:
    """`prod` 模式：断言"记录但不抛"，能跑完整条链路。理由同 `test_agent_graph.py`。"""
    return Settings(env=env, deepseek_api_key=None)


def _runtime(llm=None, retriever=None, *, env: str = "prod") -> Runtime:
    """一颗**不加载任何模型**的依赖树。**所有测试都从这里出发。**"""
    return build_runtime(
        _settings(env),
        retriever=retriever if retriever is not None else FakeRetriever(),
        llm=llm if llm is not None else ScriptedLLM(["依据 [1]，每满一年一个月工资。"]),
    )


# --------------------------------------------------------------------------- #
# ① 包入口是懒的（这是个**结构**性质，只能用子进程测）
# --------------------------------------------------------------------------- #


def test_importing_the_package_does_not_load_the_config_machinery() -> None:
    """`import juris_agent` **不能**顺带把 `client` / 配置那一套拉进来。

    ## 为什么必须开子进程

    在测试进程里 `import juris_agent` 时，别的测试早就把 `juris_agent.client`
    塞进 `sys.modules` 了——**这个断言永远是过的**，即使用 `__init__.py`
    顶部一句 `from .client import ask` 也会过。开销和收益不对称到
    这种程度的测试，不如不写。

    ## 它盯的是什么

    `juris_agent/__init__.py` 顶部的契约是"只往 `os.environ` 里塞几个
    关键变量，别的什么都不干"（见那个模块的 docstring）。一旦有人图省事
    在顶部加一句 `from .client import ask`，`pydantic-settings` 就会
    在引导阶段被加载，`HF_ENDPOINT` 生效的时机跟着漂——那是个
    **只在某些 import 顺序下才复现的模型下载超时**。
    这条断言就是那个契约的哨兵。
    """
    code = (
        "import sys;"
        "import juris_agent;"
        # 打包成一行，避免多行字符串里出现引号嵌套。
        "bad=[m for m in ('juris_agent.client','juris_agent.config') if m in sys.modules];"
        "print('|'.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "`import juris_agent` 之后这两个模块不该在 sys.modules 里，"
        f"实际加载了：{out.stdout.strip()}"
    )


def test_from_package_import_ask_works() -> None:
    """懒导出的是**真能用**的，不是占位符。"""
    import juris_agent

    assert juris_agent.ask is not None
    assert juris_agent.JurisAgent is JurisAgent
    # 第二次访问走的是普通属性查找（`__getattr__` 里塞回了 globals）。
    assert "ask" in dir(juris_agent)


def test_a_typo_is_an_attribute_error_not_a_silent_guess() -> None:
    """名字打错就该报错。**门面最怕的就是"猜"**——猜对了看着正常，
    猜错了给的是一个**同名的、别的东西**，比报错难查得多。"""
    import juris_agent

    with pytest.raises(AttributeError, match="has no attribute"):
        juris_agent.aks  # noqa: B018


# --------------------------------------------------------------------------- #
# ② 一问一答：门面不改行为
# --------------------------------------------------------------------------- #


def test_ask_returns_the_answer_and_the_citations() -> None:
    llm = ScriptedLLM(["依据 [1]，每满一年支付一个月工资。"])
    agent = JurisAgent(runtime=_runtime(llm))
    ans = agent.ask("被辞退有补偿吗？")

    assert ans.answer.startswith("依据 [1]")
    assert ans.ok, f"finish_reason={ans.finish_reason}"
    assert [c.index for c in ans.citations] == [1]
    assert ans.citations[0].text == "《劳动合同法》第四十七条"
    # 原文**没被截断**：`Answer` 上拿到的必须是完整的那一条。
    assert ans.citations[0].body == FakeRetriever().hits[0]["text"]


def test_citations_only_include_the_ones_that_really_exist() -> None:
    """模型编了一个不存在的编号（[9]）→ **引用列表里不能有它**。

    这是门面里最容易顺手做错的一件事：把 `citation_report` 里
    `invalid` 的那些也算进来。**算进来等于替模型圆谎**——
    用户看到"引用：《劳动合同法》第九条"会以为真有这条。
    它们仍然在 `state.citation_report.invalid` 里，要统计幻觉率去那儿取。
    """
    llm = ScriptedLLM(["依据 [1] 和 [9]，你可以要求赔偿。"])
    agent = JurisAgent(runtime=_runtime(llm))
    ans = agent.ask("被辞退有补偿吗？")

    assert [c.index for c in ans.citations] == [1]
    report = ans.state.citation_report
    assert report is not None
    assert 9 in report.invalid


def test_out_of_scope_has_no_citations_and_says_why_in_text() -> None:
    """超纲问题：拒答、零引用，而 `text` 要说明"这类问题不查法条"。

    留一行说明比留白好——留白看着像"检索挂了"。
    """
    agent = JurisAgent(runtime=_runtime(ScriptedLLM()))
    ans = agent.ask("离婚时财产怎么分？")

    assert ans.route == "out_of_scope"
    assert ans.citations == []
    assert "不查法条" in ans.text


def test_asked_text_is_the_answer_plus_citations() -> None:
    """`text` 是给人看的整段，且**不重复**打印答案正文。"""
    llm = ScriptedLLM(["依据 [1]，每满一年一个月工资。"])
    ans = JurisAgent(runtime=_runtime(llm)).ask("被辞退有补偿吗？")
    text = ans.text

    assert text.count("每满一年一个月工资") == 1
    assert "[1] 《劳动合同法》第四十七条" in text


def test_to_dict_is_json_serializable_and_has_no_state() -> None:
    """`to_dict()` 要能 `json.dumps`。

    ⚠️ 它**不能**含 `state`：那里面挂着 `on_delta` 之类的回调对象，
    不一定能序列化；体积也是答案的几十倍。这条断言就是防着
    "有人图省事把 `vars(state)` 整个塞进去"。
    """
    llm = ScriptedLLM(["依据 [1]。"])
    ans = JurisAgent(runtime=_runtime(llm)).ask("被辞退有补偿吗？")
    data = ans.to_dict()

    assert "state" not in data
    json.dumps(data, ensure_ascii=False)  # 不抛就算过
    assert data["citations"][0]["citation"] == "《劳动合同法》第四十七条"


# --------------------------------------------------------------------------- #
# ③ 装不起来的时候：**必须是一句能看懂的话**
# --------------------------------------------------------------------------- #


def test_not_ready_carries_the_problems_and_the_recovery_step() -> None:
    """装配失败 → `NotReady`，且**原因和恢复动作都在消息里**。

    这是个和"服务带病启动"相反的设计（理由见 `client.NotReady` 的 docstring）：
    库里调用的人没有 `/api/health` 可看。所以这里断言两件事都在：
    ① 真正的那个问题；② 改完之后该怎么让它生效（`reset()`）。
    """
    broken = Runtime(
        settings=_settings(),
        sessions=SessionStore(),
        runner=None,
        problems=["检索器未装配：ImportError: No module named 'sentence_transformers'"],
    )
    agent = JurisAgent(runtime=broken)

    with pytest.raises(NotReady) as ei:
        agent.ask("被辞退有补偿吗？")

    msg = str(ei.value)
    assert "sentence_transformers" in msg
    assert "close()" in msg
    assert ei.value.problems == broken.problems


def test_health_never_raises_even_when_broken() -> None:
    """`health()` **永远不抛**——这正是它存在的理由。

    装不起来时它也要给出同样形状的 dict，这样 `juris doctor` 和
    调用方都不用写分支。
    """
    broken = Runtime(settings=_settings(), sessions=SessionStore(), problems=["缺 API key"])
    health = JurisAgent(runtime=broken).health()

    assert health["ready"] is False
    assert health["problems"] == ["缺 API key"]
    assert health["error"]  # 也给了一份异常消息，方便直接打出来


def test_ready_does_not_trigger_a_build() -> None:
    """`ready` 是**没有副作用**的那个。

    ⚠️ 它和 `health()` 的分工很容易搞混：`ready` 只回答"现在装好了吗"，
    给"要不要现在就问"的判断用。如果它也去装一次，那"检查一下"
    就等于"加载模型"——一个只想知道状态的调用会付几分钟的代价。
    """
    agent = JurisAgent(runtime=_runtime())
    assert agent.ready is True

    fresh = JurisAgent()  # 没有任何注入，也**没有 runtime**
    assert fresh.ready is False  # 只读了一眼，没去装


def test_close_throws_away_the_runtime_and_is_safe_to_call_twice() -> None:
    """`close()` 之后回到"没装"的状态；再调一次也不该炸。

    ⚠️ **这条测试刻意不去 `ask()`。** 因为 `close()` 之后那一下会走
    `build_runtime()` 的**真实**路径——也就是在这个单测进程里
    加载约 1GB 的嵌入模型、真读一次 `.env`。第一版就是这么写的，
    结果这条"便宜"的单测花了 25 秒，还顺手在下游把真模型装了一遍。
    **一个本该毫秒级的测试里出现模型加载，是它自己在说"我测的东西跑偏了"**。

    所以这里只断言"引用确实被丢掉了"这个**可达且廉价**的性质。
    "重装能成功"这件事由 `test_health_never_raises_even_when_broken`
    等几条从**另一个方向**覆盖（给一颗装好的 runtime 看它认不认）。
    """
    agent = JurisAgent(runtime=_runtime())

    assert agent.ready is True
    agent.close()
    assert agent.ready is False
    assert agent._runtime is None, "close() 之后该把引用丢掉，下次用才会重装"

    agent.close()  # 幂等：没有 runtime 时不做事，也不抛
    assert agent.ready is False


def test_context_manager_closes() -> None:
    """`with JurisAgent() as agent:` 退出时自动 close。**这是最不容易忘的用法。**"""
    agent = JurisAgent(runtime=_runtime())
    with agent as a:
        assert a is agent
        assert a.ready is True
    assert agent.ready is False


# --------------------------------------------------------------------------- #
# ④ 多轮：会话由 SessionStore 管，不在门面里另起一套
# --------------------------------------------------------------------------- #


def _prompts(llm) -> list[str]:
    """把假模型每次收到的消息**压平成一整段文本**。用于"某个词有没有进过 prompt"。"""
    return [json.dumps(msgs, ensure_ascii=False) for msgs in llm.messages]


def test_same_session_id_accumulates_history() -> None:
    """同一个 `session_id` 的第二次提问**看得到第一轮的原文**。

    ## 断言方式选得比看起来讲究

    ⚠️ **不能断言"消息条数变多了"**：实测第一版就是这么写的，然后红了——
    这个项目的 prompt 把历史**折进一个块里**，而不是拆成一条条 message，
    所以条数一直是 2，两轮一模一样。**"条数没变"看着像历史丢了，
    其实历史就在里面。** 这正是"用错了结构性质"的典型代价：测试红了，
    而它红的原因跟它想测的东西没关系。

    ⚠️ **也不能用一个"两轮共有的词"去查**（第一版想用「月薪」）：
    那个词在第一轮的**问题本身**里就有，于是第一次调用的 prompt 里也能搜到——
    **断言恒真，永远是绿的**，等于没测。

    所以用一对**只在某一边出现**的标记：第一轮的答案里有个专属词，
    第二轮的问题里有个专属词。然后断言"第一轮的那个词，
    在第二轮才进 prompt"——这一句同时排除了上面两种假阳性。
    """
    llm = ScriptedLLM(["ZZTOP 是第一轮的答案 [1]。", "那要看工龄 [1]。"])
    rt = _runtime(llm)
    agent = JurisAgent(runtime=rt)

    agent.ask("QONE 我在西安，月薪 12000", session_id="s-t1")
    first_round = [p for p in _prompts(llm)]
    assert not any("ZZTOP" in p for p in first_round), "第一轮的答案不该出现在它自己的 prompt 里"

    agent.ask("QTWO 那能赔多少？", session_id="s-t1")
    later = [p for p in _prompts(llm)][len(first_round):]
    assert any("ZZTOP" in p and "QTWO" in p for p in later), "第二轮该同时带上上一轮的问答"

    sess = rt.sessions.get_or_create("s-t1")
    assert sess.version == 2
    assert [m["role"] for m in sess.history] == ["user", "assistant", "user", "assistant"]


def test_without_session_id_two_asks_are_independent() -> None:
    """不给 id 就是无状态。**这个默认值不能悄悄变成"有状态"。**

    ⚠️ 这里**不能**断言"两次调用的消息条数一样"来证明独立性——
    条数一样也可能只是因为两轮都恰好折成同样多的 message。
    要证明的是"第一轮的东西没漏到第二轮"，所以还是查那段文本。
    """
    llm = ScriptedLLM(["ZZTOP 是第一轮的答案 [1]。", "答第二问 [1]。"])
    rt = _runtime(llm)
    agent = JurisAgent(runtime=rt)

    agent.ask("QONE 被辞退有补偿吗？")
    n_first = len(llm.messages)
    agent.ask("QTWO 那加班费呢？")

    assert rt.sessions.stats()["sessions"] == 0, "不给 session_id 就不该建会话"
    later = _prompts(llm)[n_first:]
    assert later, "第二轮至少该调一次模型"
    assert not any("ZZTOP" in p for p in later), "第一轮的答案漏进第二轮了"


def test_session_conflict_becomes_a_warning_not_an_exception() -> None:
    """写回被 CAS 拒了 → 答案照给（钱都花了），冲突进 `warnings`。

    这里**直接调 `_commit_turn`**，不绕一大圈去构造并发：要测的是
    "冲突被翻译成了什么"，而"怎么造出冲突"在
    `tests/test_api.py` 里已经测过了。硬造一个真并发只会让这条测试
    变成时快时慢的那种。
    """
    rt = _runtime()
    sess = rt.sessions.get_or_create("s-c1")
    stale_version = sess.version
    # 模拟"读完之后别人改过"：先正常提交一轮，版本号就涨上去了。
    rt.sessions.commit(sess, expected_version=stale_version, question="q1", answer="a1")

    warnings = _commit_turn(rt, sess, stale_version, "q2", "a2")

    assert len(warnings) == 1
    assert "s-c1" in warnings[0]
    # ⚠️ **没有写回**：历史里还是第一轮那两条，没有多出 q2/a2。
    assert len(sess.history) == 2


# --------------------------------------------------------------------------- #
# ⑤ 成本：`billable` 那个标志不能被抹掉
# --------------------------------------------------------------------------- #


def test_cost_falls_back_with_billable_false_when_the_llm_keeps_no_ledger() -> None:
    """假模型没有账本（`ScriptedLLM` 就没开账）→ 走退路，且**必须标不记账**。

    ⚠️ 这是**踩过坑才加的一条**（`replay` 模式下报出过一个看着完全合理的
    `¥0.138040`，而那一轮一个接口都没打）。这里的断言就是它的回归测试：
    **`billable` 是 False**，`basis` 说清楚这是个估算。
    """
    ans = JurisAgent(runtime=_runtime(ScriptedLLM(["依据 [1]。"]))).ask("被辞退有补偿吗？")

    assert ans.cost is not None
    assert ans.cost.billable is False
    assert "估算" in ans.cost.basis or "没有调用模型" in ans.cost.basis


def test_cost_uses_the_turn_ledger_when_the_llm_has_one() -> None:
    """模型**有**账本时按分项算，且 `billable=True`。

    用一个小包装给 `ScriptedLLM` 补上 `begin_turn` / `turn_usage`——
    形状和 `llm.LLMClient` 一样。**写死一个数**是刻意的：
    这条测的是"分项有没有被用上"，不是"金额算得准不准"
    （后者归 `tests/test_cost.py`）。
    """

    class LedgerLLM(ScriptedLLM):
        def __init__(self) -> None:
            super().__init__(["依据 [1]。"])
            self._usage = Usage()

        def begin_turn(self) -> None:
            self._usage = Usage(prompt_tokens=1200, completion_tokens=80)

        @property
        def turn_usage(self) -> Usage:
            return self._usage

    ans = JurisAgent(runtime=_runtime(LedgerLLM())).ask("被辞退有补偿吗？")

    assert ans.cost.billable is True
    assert ans.cost.prompt_tokens == 1200
    assert ans.cost.completion_tokens == 80


# --------------------------------------------------------------------------- #
# ⑥ 命令行
# --------------------------------------------------------------------------- #


def test_parser_builds_and_every_subcommand_is_reachable() -> None:
    """四个子命令都能解析。

    ⚠️ 这条测试盯的是一个**位置很靠后的坑**：`--session` 在两个子解析器上
    都注册的话，`add_argument` **当场就抛**
    （`conflicting option string`），但这个文件里如果没人为每个子命令
    跑一遍 `parse_args`，就没人碰得到它——**要等真有人敲
    `juris chat --session x` 才炸**。
    """
    parser = build_parser()

    assert parser.parse_args(["ask", "问题"]).question == "问题"
    assert parser.parse_args(["chat"]).session is None
    assert parser.parse_args(["chat", "--session", "s-1"]).session == "s-1"
    assert parser.parse_args(["serve"]).port == 8000
    assert parser.parse_args(["doctor"]).func.__name__ == "cmd_doctor"


def test_no_subcommand_prints_help_and_exits_nonzero(capsys) -> None:
    """不给子命令 → 打帮助，退出码 **2**（argparse 对用法错误用的就是 2）。

    **不能用 0**：那样 `juris || echo 失败了` 是哑的，
    而"敲错命令但脚本继续往下跑"是最容易酿成事故的一类静默失败。
    """
    assert main([]) == 2
    assert "<命令>" in capsys.readouterr().err


def test_ask_without_a_question_exits_two(monkeypatch) -> None:
    """`juris ask -` 且标准输入是空的 → 用法错误（2），不是崩。

    ⚠️ **必须把 `sys.stdin` 换掉。** pytest 会拦截真实的标准输入
    （`DontReadFromInput`），碰一下就抛 `OSError: reading from stdin while
    output is captured`——**那条报错说的是测试环境的事，
    和被测代码一点关系没有**，第一次看到很容易以为是自己代码写错了。
    换成一个空的 `StringIO`，测的就回到"读到空串之后怎么办"这个真问题上了。
    """
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["ask", "-"]) == 2


def test_unknown_subcommand_is_a_usage_error() -> None:
    """瞎敲一个子命令：argparse 自己会 `SystemExit(2)`。**以退出码为准。**"""
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(["frobnicate"])
    assert ei.value.code == 2


def test_cli_sends_all_logging_to_stderr(monkeypatch) -> None:
    """命令行的日志**必须走 stderr**，否则 `--json | jq` 当场就废。

    ## 这条断言盯的是一个"在别处坏掉"的约定

    `observability/logs.py` 的默认流是 **stdout**（服务端要的就是
    "日志和业务输出混在一起给日志平台收"）。命令行刚好相反。
    所以 `cli._setup_logging` 显式传 `stream=sys.stderr`。

    ⚠️ **实测过一次真的坏了**：`configure_logging` 里有"配过就不再配"
    的开关，而 `from juris_agent.client import ...` 会顺手把日志配成
    默认的 stdout 版。于是如果 `_setup_logging` 调用得不够早，
    它的诉求被**静默忽略**，httpx 的 `HTTP Request:` 和 jieba 的
    "Building prefix dict" 全进了 stdout——存下来的 `answer.txt` 里
    混着几行时间戳，`--json` 那一份也不再是合法 JSON 了。

    ## 怎么测才不会误伤别的测试

    ⚠️ **不能真调 `configure_logging`**：它会 `logging.basicConfig`
    装一个 handler 到根日志器上，然后**整个 pytest 进程**的日志
    就都跟着变了（而且 `_CONFIGURED` 这个全局开关会被置上，
    后面别的测试再也配不回来）。所以这里把它换成一个记账的假函数，
    断言**传进去的参数**——要测的正是"传了 stderr 这个决定"。

    `_quiet_third_party` 也一并换掉：它里面要 import
    `sentence_transformers`（**几秒 + torch**），为了断言
    "它有没有被调用"而在单测里装一遍 torch 是极不划算的。
    """
    from juris_agent import cli, observability

    seen: dict[str, object] = {}

    def fake_configure(**kw: object) -> None:
        seen.update(kw)

    quieted: list[bool] = []
    monkeypatch.setattr(observability, "configure_logging", fake_configure)
    monkeypatch.setattr(cli, "_quiet_third_party", lambda: quieted.append(True))

    cli._setup_logging(verbose=False)

    assert seen["stream"] is sys.stderr, "日志没走 stderr，stdout 会被日志污染"
    assert seen["level"] == "WARNING"
    assert quieted == [True], "非 verbose 时该把第三方日志按下去"

    # verbose=True：**只放开级别，流还是 stderr**。
    # 这两件事是独立的：想留日志的人该做 `2> log.txt`，
    # 而不是让答案和日志混在同一个流里。
    seen.clear()
    quieted.clear()
    cli._setup_logging(verbose=True)

    assert seen["stream"] is sys.stderr
    assert seen["level"] == "INFO"
    assert quieted == [], "verbose 时不该按掉任何东西"


def test_stream_mode_still_prints_the_answer_when_nothing_was_streamed(capsys) -> None:
    """`--stream` 但**一个增量都没收到**时，正文不能被弄丢。

    ## 这条盯的是一个真写出来过的 bug

    `on_delta` **不是每轮都会被调**：只有"生成节点调模型"那条路会推增量。
    拒答、闲聊、超纲、以及没有模型时，正文是节点**直接写进状态**的。

    第一版的 `_answer_of` 只看 `args.stream`——于是这些路由下
    正文一个字都不打，只打出一行「引用法条：」。
    用户看到的是**一个空回答**，而且
    `juris ask --stream "离婚时财产怎么分"` 和 `juris ask "离婚时财产怎么分"`
    **对同一个问题给出不同的东西**。

    ## 为什么用假 agent 而不是真跑一遍

    要构造的是"路由直接写答案、不走模型"这个**形状**，而触发它
    （挑一个超纲问题）需要真模型。假 agent 把这个形状直接摆出来，
    测试因此是确定的、毫秒级的——和 `tests/fakes.py` 整个文件
    的立场一致：**真模型测效果，假件测逻辑。**
    """

    class _NoStreamAgent:
        """一个**永远不调 `on_delta`** 的 agent：正文在状态里，没有增量。"""

        def ask(self, question, *, session_id=None, on_delta=None):
            return Answer(
                question=question,
                answer="这类问题不归劳动法管，我不答。",
                citations=[Citation(index=1, key="lcl#1", text="《劳动合同法》第一条")],
                route="out_of_scope",
                finish_reason="refused",
            )

    args = build_parser().parse_args(["ask", "离婚时财产怎么分？", "--stream"])
    ans, streamed = _answer_of(_NoStreamAgent(), "离婚时财产怎么分？", args)

    assert streamed is False, "没收到增量就不该说正文已经打过了"
    _print_answer(ans, args, streamed)

    out = capsys.readouterr().out  # ⚠️ 只取 stdout：诊断行走的是 stderr
    assert "这类问题不归劳动法管" in out, "正文被弄丢了——只打了引用"
    assert "《劳动合同法》第一条" in out
    assert out.count("这类问题不归劳动法管") == 1, "正文打了两遍"
    assert ans.answer.startswith("这类问题")


def test_stream_mode_prints_the_body_once_when_deltas_arrive(capsys) -> None:
    """收到增量时：正文靠增量打（一次），**引用另外补**，正文不重复。"""

    class _StreamingAgent:
        """真会推增量的 agent：正文由 `on_delta` 交出去。"""

        def ask(self, question, *, session_id=None, on_delta=None):
            text = "每满一年一个月工资。"
            assert on_delta is not None, "开了 --stream 就该把回调传下来"
            for ch in text:  # 逐字推，模拟打字机
                on_delta(ch)
            return Answer(
                question=question,
                answer=text,
                citations=[Citation(index=1, key="lcl#47", text="《劳动合同法》第四十七条")],
                route="law",
                finish_reason="answered",
            )

    args = build_parser().parse_args(["ask", "被辞退有补偿吗？", "--stream"])
    ans, streamed = _answer_of(_StreamingAgent(), "被辞退有补偿吗？", args)

    assert streamed is True
    _print_answer(ans, args, streamed)

    out = capsys.readouterr().out
    assert out.count("每满一年一个月工资") == 1, f"正文打了 {out.count('每满一年一个月工资')} 遍"
    assert "《劳动合同法》第四十七条" in out


def test_json_output_is_pure_json() -> None:
    """`--json` 的输出**必须整段可以被 `json.loads`**。

    这是"别把人话和 JSON 混着打"那条规矩的验收：混着打的话，
    调用方得先做字符串切分才知道从哪儿开始是 JSON，
    而**切分逻辑迟早跟不上格式变化**。
    """
    from juris_agent.cli import _emit_json

    ans = Answer(
        question="q",
        answer="a",
        citations=[Citation(index=1, key="lcl#47", text="《劳动合同法》第四十七条")],
        route="law",
        finish_reason="answered",
    )
    # 用一个真文件当 stdout：capsys 也能用，但这里要的是"整段合法"，
    # 直接抓字符串更直白。
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_json(ans)

    assert json.loads(buf.getvalue())["answer"] == "a"
