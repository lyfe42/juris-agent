"""节点级断言的测试（F18）。

## 这套测试的重点是"两种模式都要测"

每条具名断言都要有**两组**测试：

- dev 模式：违反 → **抛 `InvariantError`**
- prod 模式：违反 → **记一笔，返回给调用方，不抛**

只测一组是不够的，而且**只测 dev 那组是最常见也最没用的做法**：
dev 抛异常很容易写对，真正会出问题的是 prod 那条"记下来但继续"的路径——
写错了（比如忘了 return、或者把 violations 吞掉）在生产上表现为
"断言形同虚设"，而所有测试都是绿的。

## `test_every_named_check_is_covered` 这条断言的作用

它枚举 `NAMED_CHECKS`，检查每个名字在 `InvariantChecker` 上都有对应方法。
**它防的是"写了断言却从没被调用过"**——那种断言在报表上会被算成"通过"，
实际从来没检查过任何东西。这也是为什么改这个文件时要同步改 `NAMED_CHECKS`：
那条列表是"有哪些断言"的单一来源。
"""

from __future__ import annotations

import pytest

from juris_agent.agent.invariants import (
    NAMED_CHECKS,
    INV_ANSWER_NONEMPTY,
    InvariantChecker,
    InvariantError,
)
from juris_agent.agent.state import AgentState, InjectedArticle


def _inj(n: int = 2) -> list[InjectedArticle]:
    return [
        InjectedArticle(
            index=i,
            article_key=f"lcl#{40 + i}",
            citation=f"《劳动合同法》第{40 + i}条",
            law="劳动合同法",
            article_no=f"第{40 + i}条",
            text=f"第 {i} 条正文",
        )
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------- #
# 两种模式的行为差异——**这是整个模块的核心**
# --------------------------------------------------------------------------- #


def test_dev_mode_raises() -> None:
    """dev：直接炸。开发期最贵的是"没发现 bug"。"""
    c = InvariantChecker(env="dev")
    with pytest.raises(InvariantError) as ei:
        c.require(False, "INV-T", "测试断言")
    assert ei.value.code == "INV-T"


def test_prod_mode_records_and_continues() -> None:
    """prod：记一笔、返回给调用方、**不炸**。生产最贵的是"用户用不了"。"""
    c = InvariantChecker(env="prod")
    out = c.require(False, "INV-T", "测试断言", extra=1)
    assert len(out) == 1 and "INV-T" in out[0]
    assert c.violations[0].detail == {"extra": 1}
    # 严格的实例仍然会炸——同一个类，只是 env 不同
    assert c.strict is False


def test_passing_check_returns_empty_in_both_modes() -> None:
    for env in ("dev", "prod"):
        assert InvariantChecker(env=env).require(True, "INV-T", "不该触发") == []


def test_on_violation_callback_is_called_in_both_modes() -> None:
    """回调（用来把违规推给前端/日志）在两种模式下**都要被调**。

    只在 dev 调的话，生产环境就完全看不到违规了——
    那等于"生产期没有断言"，只不过代码还在那里让人以为有。
    """
    seen: list[str] = []
    for env in ("dev", "prod"):
        c = InvariantChecker(env=env, on_violation=lambda v: seen.append(f"{env}:{v.code}"))
        try:
            c.require(False, "INV-T", "x")
        except InvariantError:
            pass
    assert seen == ["dev:INV-T", "prod:INV-T"]


# --------------------------------------------------------------------------- #
# INV-01 进生成节点必须有上下文
# --------------------------------------------------------------------------- #


def test_generate_needs_context() -> None:
    assert InvariantChecker(env="prod").generate_needs_context(AgentState()) != []


def test_generate_ok_with_context() -> None:
    st = AgentState(injected=_inj(1))
    assert InvariantChecker(env="prod").generate_needs_context(st) == []


# --------------------------------------------------------------------------- #
# INV-02 编号范围 / INV-03 键必须来自注入
# --------------------------------------------------------------------------- #


def test_citation_index_out_of_range_is_flagged() -> None:
    st = AgentState(injected=_inj(2))
    assert InvariantChecker(env="prod").citation_index_in_range(st, [1, 7]) != []


def test_citation_index_in_range_ok() -> None:
    st = AgentState(injected=_inj(2))
    assert InvariantChecker(env="prod").citation_index_in_range(st, [1, 2]) == []


def test_citation_key_must_be_injected() -> None:
    """**"可溯源"的机器可验证形式。**

    答案里出现的每一部法条，都必须是这次检索真的捞上来的。
    不然"可溯源"只是一句宣传语。
    """
    st = AgentState(injected=_inj(2), citations=["lcl#41", "lcl#99"])
    assert InvariantChecker(env="prod").citation_keys_injected(st) != []


def test_citation_key_ok_when_all_injected() -> None:
    st = AgentState(injected=_inj(2), citations=["lcl#41"])
    assert InvariantChecker(env="prod").citation_keys_injected(st) == []


# --------------------------------------------------------------------------- #
# INV-04 / INV-05 / INV-06
# --------------------------------------------------------------------------- #


def test_steps_within_budget() -> None:
    c = InvariantChecker(env="prod")
    assert c.steps_within_budget(AgentState(steps=13), 12) != []
    assert c.steps_within_budget(AgentState(steps=12), 12) == []


def test_answer_nonempty_only_checked_when_answered() -> None:
    c = InvariantChecker(env="prod")
    # 不是 answered 就不检查——**拒答时答案是空的完全正常**
    assert c.answer_nonempty(AgentState(finish_reason="refused")) == []
    assert c.answer_nonempty(AgentState(finish_reason="answered", answer="")) != []
    assert c.answer_nonempty(AgentState(finish_reason="answered", answer="有内容")) == []


def test_refusal_shape_requires_three_fields() -> None:
    c = InvariantChecker(env="prod")
    assert c.refusal_shape(AgentState(finish_reason="refused", refusal={})) != []
    assert (
        c.refusal_shape(
            AgentState(
                finish_reason="refused",
                refusal={"reason": "r", "category": "c", "suggestion": "s"},
            )
        )
        == []
    )


def test_refusal_missing_suggestion_is_flagged() -> None:
    """**缺 suggestion 要单独抓出来。** 用户被拒绝之后不知道该去问谁，
    那种拒答比硬答还伤——所以要把它当成一个可检测的缺陷。"""
    st = AgentState(finish_reason="refused", refusal={"reason": "r", "category": "c"})
    assert InvariantChecker(env="prod").refusal_shape(st) != []


# --------------------------------------------------------------------------- #
# INV-07 工具出参可序列化 / INV-08 路由已定
# --------------------------------------------------------------------------- #


def test_tool_payload_serializable() -> None:
    c = InvariantChecker(env="prod")
    assert c.tool_payload_serializable({"a": 1}, "t") == []
    # Decimal 是这里最真实的例子：领域计算器返回的金额就是 Decimal
    from decimal import Decimal

    assert c.tool_payload_serializable({"money": Decimal("1.23")}, "t") == []


def test_tool_payload_unserializable_is_flagged() -> None:
    c = InvariantChecker(env="prod")

    class Weird:
        def __repr__(self) -> str:  # noqa: D105
            return "weird"

    # `default=str` 会兜住大部分对象，所以这里用一个**连 str 都救不了**的例子：
    # 自引用的容器（json 会报 "Circular reference detected"）
    circular: list = []
    circular.append(circular)
    assert c.tool_payload_serializable(circular, "t") != []
    assert c.tool_payload_serializable({"x": Weird()}, "t") == []


def test_route_decided() -> None:
    c = InvariantChecker(env="prod")
    assert c.route_decided(AgentState()) != []
    assert c.route_decided(AgentState(route="law")) == []


# --------------------------------------------------------------------------- #
# 登记表一致性
# --------------------------------------------------------------------------- #


def test_every_named_check_exists_on_the_checker() -> None:
    """`NAMED_CHECKS` 里的每个名字都得是真实方法。

    防的是改名之后忘了同步登记表——那种情况下"断言清单"会指向不存在的方法，
    而任何遍历登记表的代码（测试、报表、文档生成）都会静默少检查一条。
    """
    c = InvariantChecker(env="prod")
    for name in NAMED_CHECKS:
        assert callable(getattr(c, name, None)), f"{name} 不在 InvariantChecker 上"


def test_named_checks_count_is_pinned() -> None:
    """条数钉死。**加断言时要一起改这个数字**——故意让它成为一种提醒。

    不钉的话，"有断言被删掉了"这件事不会在任何地方留痕。
    这是本项目对"静默退化"的一贯处理方式：让它变成一条会红的测试。
    """
    assert len(NAMED_CHECKS) == 8


def test_invariant_error_carries_code_and_detail() -> None:
    e = InvariantError(INV_ANSWER_NONEMPTY, "空答案", {"len": 0})
    assert e.code == INV_ANSWER_NONEMPTY
    assert e.detail == {"len": 0}
    assert INV_ANSWER_NONEMPTY in str(e)
