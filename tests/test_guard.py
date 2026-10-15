"""四道防死循环闸门的测试（F17）。

## 这些测试真正想证明的东西

不是"闸门能被触发"（那很容易），而是三件更难的事：

1. **四道闸门互相独立。** 每道都要有一条测试，把其他三道都设置成"不可能触发"，
   只让目标那道触发。否则你会得到"某个测试过了，但不知道是哪道闸门拦住的"。
2. **边界是 `>=` 不是 `>`。** 差一步的语义差别会被参数化测出来。
3. **"连续"是连续，不是累计。** 中间夹一次别的调用，计数必须归零——
   否则正常的"第 1 步查了、第 5 步再查一次"会被误杀。

## 假时钟/假随机在这里不需要

闸门是**纯函数**（读状态、返回判定），没有时间依赖、没有随机。
所以这些测试跑在微秒级，也不需要 `conftest.py` 里的 `FakeClock`。
**纯函数不用假件，是设计对了的信号。**
"""

from __future__ import annotations

import pytest

from juris_agent.agent.guard import (
    LIMIT_BUDGET,
    LIMIT_NO_PROGRESS,
    LIMIT_REPEAT,
    LIMIT_STEPS,
    args_hash,
    call_signature,
    check,
    no_progress_run,
    progress_signature,
    result_digest,
    snapshot,
    strip_echo,
    trailing_run,
)
from juris_agent.agent.state import AgentState, InjectedArticle, ToolCallRecord
from juris_agent.config import GuardSettings


@pytest.fixture
def g() -> GuardSettings:
    """小参数的闸门，边界好构造（默认值 12/2/3/60000 要造很多数据）。"""
    return GuardSettings(
        max_steps=4, repeat_call_limit=2, no_progress_limit=3, token_budget=1000
    )


def _rec(tool: str = "search_law", args: dict | None = None, status: str = "ok") -> ToolCallRecord:
    return ToolCallRecord(step=1, tool=tool, args=args or {"query": "a"}, status=status)


def _ctx() -> list[InjectedArticle]:
    return [
        InjectedArticle(index=1, article_key="lcl#47", citation="《劳动合同法》第四十七条",
                        law="劳动合同法", article_no="第四十七条", text="……")
    ]


# --------------------------------------------------------------------------- #
# 小工具：trailing_run / args_hash
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "items,expected",
    [
        ([], 0),
        (["a"], 1),
        (["a", "b"], 1),
        (["a", "b", "b"], 2),
        (["b", "b", "b"], 3),
        (["a", "a", "b"], 1),
    ],
)
def test_trailing_run(items: list[str], expected: int) -> None:
    assert trailing_run(items) == expected


def test_args_hash_is_order_insensitive() -> None:
    """**键顺序不同必须算出同一个哈希。**

    否则模型把 `{"a":1,"b":2}` 写成 `{"b":2,"a":1}` 就绕过了重复检测——
    闸门形同虚设，而且是**静默**失效（报表上"重复调用 0 次"，看起来很好）。
    """
    assert args_hash({"a": 1, "b": 2}) == args_hash({"b": 2, "a": 1})


def test_args_hash_distinguishes_values() -> None:
    """值不同必须算出不同哈希——否则"改了参数重试"会被误判成"重复调用"。"""
    assert args_hash({"months": 3}) != args_hash({"months": 30})


def test_call_signature_includes_tool_and_args() -> None:
    assert call_signature(_rec("a", {"x": 1})) != call_signature(_rec("b", {"x": 1}))
    assert call_signature(_rec("a", {"x": 1})) != call_signature(_rec("a", {"x": 2}))


# --------------------------------------------------------------------------- #
# ① 最大步数
# --------------------------------------------------------------------------- #


def test_steps_limit_trips_at_the_boundary(g: GuardSettings) -> None:
    """`>=` 而不是 `>`：`max_steps=4` 时第 4 步**就该停**，不是第 5 步。

    差这一步会让 max_steps 的实际语义变成"max_steps+1"，报表里报的
    "平均步数 4.0"和"上限 4 步"会自相矛盾。
    """
    assert check(AgentState(steps=3), g).ok
    d = check(AgentState(steps=4), g)
    assert not d.ok and d.limit == LIMIT_STEPS


def test_steps_limit_wins_over_others(g: GuardSettings) -> None:
    """多道闸门同时满足时，报**第一道**（步数）。顺序即优先级，写死在 `check()` 里。"""
    st = AgentState(steps=99, tokens_used=10**9, progress_keys=["-"] * 9,
                    tool_results=[_rec()] * 9)
    assert check(st, g).limit == LIMIT_STEPS


# --------------------------------------------------------------------------- #
# ② 重复调用
# --------------------------------------------------------------------------- #


def test_repeat_call_trips_on_consecutive_identical_calls(g: GuardSettings) -> None:
    st = AgentState(steps=2, tool_results=[_rec(args={"q": "x"}), _rec(args={"q": "x"})])
    d = check(st, g)
    assert not d.ok and d.limit == LIMIT_REPEAT
    assert d.detail["consecutive"] == 2


def test_repeat_call_resets_when_a_different_call_intervenes(g: GuardSettings) -> None:
    """**"连续"是连续，不是累计**——这条是闸门最容易写错的地方。

    `[A, B, A]` 里 A 出现了两次，但它们**不连续**。中间那次 B 说明
    模型在往前走（换了个工具/换了参数），这时把 A 算成"重复"会误杀。
    """
    st = AgentState(
        steps=3,
        tool_results=[_rec(args={"q": "x"}), _rec(args={"q": "y"}), _rec(args={"q": "x"})],
    )
    assert check(st, g).ok, "A-B-A 不该被判成重复调用"


def test_repeat_call_counts_only_identical_args(g: GuardSettings) -> None:
    """同名工具、不同参数 → 不算重复。这正是"改参数重试"的正常路径。"""
    st = AgentState(steps=2, tool_results=[_rec(args={"q": "x"}), _rec(args={"q": "z"})])
    assert check(st, g).ok


# --------------------------------------------------------------------------- #
# ③ 无进展
# --------------------------------------------------------------------------- #


def test_no_progress_trips_after_limit(g: GuardSettings) -> None:
    st = AgentState(steps=3, progress_keys=["-", "-", "-"])
    d = check(st, g)
    assert not d.ok and d.limit == LIMIT_NO_PROGRESS
    assert d.detail["consecutive"] == 3


def test_no_progress_does_not_trip_below_limit(g: GuardSettings) -> None:
    assert check(AgentState(steps=2, progress_keys=["-", "-"]), g).ok


def test_progress_with_new_keys_resets_the_count(g: GuardSettings) -> None:
    """检索到**新**条法条 = 有进展，即使前面已经打转两步。"""
    st = AgentState(
        steps=3,
        progress_keys=["-", "-", progress_signature(new_keys=["lcl#47"])],
    )
    assert check(st, g).ok


def test_tool_failure_still_counts_as_progress() -> None:
    """⚠️ **工具失败也是进展。**

    这个选择是有代价的（"一直失败一直重试"不会被无进展闸门拦住），
    但反过来更糟：把失败算成没进展，会让"工具坏了、Agent 正在如实报告"
    这条完全正常的轨迹在第 3 步被掐断。
    **拦住"一直失败"是②重复调用和①步数上限的活，不是③的活。**
    """
    assert progress_signature(tool_results=[_rec(status="failed")]) != "-"


def test_progress_signature_is_order_insensitive_for_keys() -> None:
    """同样 8 条、顺序不同 → 同一个指纹。

    否则模型每轮重新检索都能"刷出"一个不同的指纹，无进展闸门就永远不触发。
    """
    a = progress_signature(new_keys=["a", "b", "c"])
    b = progress_signature(new_keys=["c", "a", "b"])
    assert a == b


def test_progress_signature_marks_answer() -> None:
    assert progress_signature(produced_answer=True) == "answer"


def test_empty_progress_is_the_sentinel() -> None:
    assert progress_signature() == "-"


# --------------------------------------------------------------------------- #
# ③ 的两种"停住了"的形态——**这一段是踩坑之后补的**
#
# 说明为什么 `no_progress_run` 不能写成"数末尾连续相同"，以及 `strip_echo` 是干嘛的：
# 它们的单测就是"这两个坑不会再踩一遍"的证据。
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        # 全是新的 → 每一步都算进展
        (["a", "b", "c"], 0),
        # ⚠️ **这一组是重点**：相邻都不同（"末尾连续相同"只数到 1），
        #    但它就是停的——闸门必须在这里也能看出来
        (["a", "b", "a", "b"], 2),
        # 往回数到第二项就断了：`items[1]` 的 "b" 是头一次出现
        (["a", "b", "a", "b", "a", "b"], 4),
        # 第一个 a 是新的（那一步是进展），后两个才是白跑
        (["a", "a", "a"], 2),
        (["x", "a", "a", "a"], 2),
        # 空产出永远是"没进展"，不需要"以前见过"这个条件
        (["-", "-", "-"], 3),
        (["a", "-", "a"], 2),  # 末尾两步：空产出 + a 是老面孔
    ],
)
def test_no_progress_run(items: list[str], expected: int) -> None:
    """③号闸门用的读数。**这两条口径要一起记住**：

    1. "以前见过的东西"算没进展 → 抓的是"换个说法来回问"（第二种原地打转）
    2. "什么都没产出"算没进展，**且不需要以前见过** → 抓的是"空转"

    第一版只有口径 1 的半个（只看末尾连续相同），上面 `[a,b,a,b]` 那组
    只会数到 1，于是最常见的原地打转**永远拦不住**，一路烧到步数上限。
    """
    assert no_progress_run(items) == expected


def test_no_progress_run_is_stricter_about_the_first_of_a_kind() -> None:
    """同一个 `[a,a,a]`，两个计数函数给出的数**故意不同**：

    - `trailing_run` = 3（"末尾三个都是 a"）
    - `no_progress_run` = 2（第一个 a 是新的，那步算进展）

    这两个数**不能混用**：②号闸门（重复调用）要的是前者，
    因为"调了两次一模一样的"就该在第 2 次掐；③号闸门要的是后者。
    """
    assert trailing_run(["a", "a", "a"]) == 3
    assert no_progress_run(["a", "a", "a"]) == 2


# ---- 回声字段（`search_law` 把 query 抄回来那个坑） ------------------------- #


def _search_rec(query: str, hits: list[str] | None = None) -> ToolCallRecord:
    """造一条 `search_law` 的调用记录。**入参里的 query 会被出参原样抄回来**——
    真实的 `LawSearchResult` 就是这么设计的（方便模型对账），
    这个"抄回来"正是把无进展闸门骗过去的东西。"""
    return ToolCallRecord(
        step=1,
        tool="search_law",
        args={"query": query, "top_k": 8},
        status="ok",
        payload={
            "query": query,
            "hits": [{"article_key": k, "text": "……"} for k in (hits or ["lcl#46", "lcl#47"])],
            "citations": ["《劳动合同法》第四十六条", "《劳动合同法》第四十七条"],
            "warnings": [],
        },
    )


def test_strip_echo_only_erases_the_matching_string_field() -> None:
    """回声字段被抹掉，**别的字段一个字都不动**。"""
    out = strip_echo(
        {"query": "同一句话", "note": "别的话", "top_k": 8, "nested": {"query": "同一句话"}},
        {"query": "同一句话", "top_k": 8},
    )
    assert out["query"] != "同一句话"  # 回声 → 被抹
    assert out["note"] == "别的话"      # 值不一样 → 保留
    assert out["top_k"] == 8           # **只处理字符串**：数字字段不动
    assert out["nested"]["query"] != "同一句话"  # 嵌套也要走一遍


def test_result_digest_ignores_the_echoed_query() -> None:
    """换个说法问同一件事 → **同一个指纹**。"""
    a = result_digest(_search_rec("经济补偿 计算标准").payload, {"query": "经济补偿 计算标准"})
    b = result_digest(_search_rec("解除劳动合同 补偿").payload, {"query": "解除劳动合同 补偿"})
    assert a == b, "query 是回声，不该影响指纹——否则模型换个说法就能骗过闸门"


def test_result_digest_notices_different_hits() -> None:
    a = result_digest(_search_rec("q", hits=["lcl#47"]).payload, {"query": "q"})
    b = result_digest(_search_rec("q", hits=["lcl#87"]).payload, {"query": "q"})
    assert a != b, "真的查到了不同的法条 → 必须算进展"


def test_progress_signature_ignores_the_echoed_query() -> None:
    """把上面两件事合起来：**换个说法重查一遍，闸门看到的是"没进展"**。"""
    assert progress_signature(tool_results=[_search_rec("甲")]) == progress_signature(
        tool_results=[_search_rec("乙")]
    )


def test_no_progress_trips_on_rephrased_repeats(g: GuardSettings) -> None:
    """端到端：模型换了四种说法、每次返回同一批法条 → 撞③号闸门。

    **这是"原地打转"在生产里最常见的样子。** 第一版代码在这里会被骗过去：
    参数每次不同 → ②不触发；指纹每次不同 → ③也不触发 → 一路烧到 12 步上限。

    四次要的是"连续三次没进展"（`g` 里 `no_progress_limit=3`；
    第一次拿回来的东西是新的，那一步不算）。
    """
    recs = [
        _search_rec("第一次问"),
        _search_rec("第二次问"),
        _search_rec("换种说法再问"),
        _search_rec("再换个说法"),
    ]
    st = AgentState(
        steps=3,  # 别撞上①步数上限（`g` 里是 4），这里只想隔离出③
        tool_results=recs,
        progress_keys=[*[progress_signature(tool_results=[r]) for r in recs]],
    )
    d = check(st, g)
    assert not d.ok and d.limit == LIMIT_NO_PROGRESS
    assert d.detail["consecutive"] == 3


def test_tool_failure_vectors_are_not_confused_with_success() -> None:
    """失败和成功**必须指纹不同**——否则"工具坏了"和"工具算出来了"混成一个。"""
    ok = _search_rec("q")
    bad = _search_rec("q").model_copy(update={"status": "failed"})
    assert progress_signature(tool_results=[ok]) != progress_signature(tool_results=[bad])


# --------------------------------------------------------------------------- #
# ④ token 预算
# --------------------------------------------------------------------------- #


def test_budget_trips_above_limit(g: GuardSettings) -> None:
    d = check(AgentState(steps=1, tokens_used=1001), g)
    assert not d.ok and d.limit == LIMIT_BUDGET


def test_budget_at_limit_is_ok(g: GuardSettings) -> None:
    """边界：正好等于预算**不算超**（`>` 而不是 `>=`）。

    和①刻意不同：步数是"离散的资源份额，用完就没了"，预算是一个上界。
    两者语义不同，所以比较符也不同——**这种不一致是有意的，注释写清楚，
    免得以后有人"顺手统一"**。
    """
    assert check(AgentState(steps=1, tokens_used=1000), g).ok


# --------------------------------------------------------------------------- #
# 快照（给界面/报表用）
# --------------------------------------------------------------------------- #


def test_snapshot_reports_all_four_readings(g: GuardSettings) -> None:
    st = AgentState(steps=2, tokens_used=50, progress_keys=["-"], tool_results=[_rec()])
    s = snapshot(st, g)
    assert s["steps"] == 2 and s["steps_left"] == 2
    assert s["consecutive_same_call"] == 1
    assert s["consecutive_no_progress"] == 1
    assert s["token_budget"] == 1000


def test_snapshot_steps_left_never_negative(g: GuardSettings) -> None:
    assert snapshot(AgentState(steps=99), g)["steps_left"] == 0
