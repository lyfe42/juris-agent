"""成本记账（`observability/cost.py`）的测试。

## 为什么这一层值得单独一套测试

它是**唯一一个"错了也没人会发现"的模块**：成本算错了，接口照样 200、
答案照样正确、测试照样全绿。唯一的症状是报表上一个偏大或偏小的数，
而那个数**看着永远很合理**——`¥0.138040` 这种数字，谁也不会怀疑它。

所以这里的断言不测"函数被调用了"，而是测**那几件容易搞混的事**：

| 容易搞混的 | 后果 | 本文件里对应的测试 |
|---|---|---|
| 思维链重复计价 | 输出成本翻倍 | `test_reasoning_tokens_are_not_charged_twice` |
| 没给命中/未命中拆分时的取舍 | 报便宜了 → 以为还有预算 | `test_missing_split_is_charged_as_miss` |
| 查不到牌价的模型 | 用一个编出来的价算钱 | `test_unknown_model_costs_zero_not_a_guess` |
| "没花钱" vs "没记账" vs "重放" | **报表虚报开销** | `test_describe_cost_separates_the_three_kinds_of_zero` |
| 汇总时丢掉 `billable` | 离线跑一遍被算成要钱 | `test_sum_propagates_billable` |

最后一条是**踩过坑才有的**：`replay` 模式跑演示时，终端和网页都打出了
`¥0.138040`，而那一轮一个接口都没打。钱的来源是"状态里记着当初真调用掉的
token 数"，公式完全正确、数据完全真实——**只是问的是一年前的问题**。
"""

from __future__ import annotations

import pytest

from juris_agent.observability import (
    CostBreakdown,
    describe_cost,
    estimate_cost,
    sum_costs,
)


# --------------------------------------------------------------------------- #
# 输入 token 的三种形态（`estimate_cost` docstring 里列的那三种）
# --------------------------------------------------------------------------- #


def test_missing_split_is_charged_as_miss() -> None:
    """没给命中/未命中拆分时，**全部按未命中算**（贵的那个）。

    这是刻意的保守。两个方向的错误代价不对称：
    报贵了会被复查的人抓出来"你算多了"；报便宜了会让人以为"成本没超"
    而继续加功能、继续扩评测集。**宁可被骂贵，不可被骂假。**
    """
    c = estimate_cost(model="deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0)
    assert c.cache_miss_tokens == 1_000_000
    assert c.cache_hit_tokens == 0


def test_given_split_is_respected() -> None:
    """给了两个数就按给的算，不覆盖。"""
    c = estimate_cost(
        model="deepseek-flash",
        prompt_tokens=1_000_000,
        cache_hit_tokens=900_000,
        cache_miss_tokens=100_000,
    )
    assert (c.cache_hit_tokens, c.cache_miss_tokens) == (900_000, 100_000)


def test_partial_split_fills_the_gap_from_the_total() -> None:
    """只给了命中数时，缺的那个按 `总输入 - 命中` 补，**不是**当成 0。

    当成 0 的话，剩下那 90 万 token 就一分钱不算了——成本表会凭空漏掉一大块，
    而且漏的方式是"某个模型特别便宜"，看着像个好消息。
    """
    c = estimate_cost(model="deepseek-flash", prompt_tokens=1_000_000, cache_hit_tokens=100_000)
    assert c.cache_hit_tokens == 100_000
    assert c.cache_miss_tokens == 900_000, "缺口没补上，那一部分没算钱"


def test_reasoning_tokens_are_not_charged_twice() -> None:
    """**思维链不单独计价**——它已经包含在 `completion_tokens` 里了。

    供应商就是这么算的。再乘一次输出单价，输出成本直接翻倍，
    而症状是"推理模型比预想的贵一倍"——听起来完全合理，
    所以这个 bug 能活很久。
    """
    with_reasoning = estimate_cost(
        model="deepseek-flash", completion_tokens=1000, reasoning_tokens=900
    )
    without = estimate_cost(model="deepseek-flash", completion_tokens=1000)
    assert with_reasoning.total_cny == without.total_cny
    assert with_reasoning.reasoning_tokens == 900, "单独记下来还是要的，报表要回答'钱花哪了'"


def test_unknown_model_costs_zero_and_says_so() -> None:
    """牌价表里没有的模型 → **成本 0，而不是编一个平均价**。

    编一个价的危害：那个假数字会和一串真数字并排显示在报表里，
    **没有任何人看得出来哪一行是编的**。报 0 则一眼可见（同列全是 0）。
    """
    c = estimate_cost(model="一个牌价表里没有的模型", prompt_tokens=10_000, completion_tokens=1_000)
    assert c.total_cny == 0.0
    assert c.prompt_tokens == 10_000, "token 数照记，只是没价钱可乘"
    assert c.basis, "金额为 0 时必须留下出处，否则和'没记账'分不开"


def test_basis_carries_the_price_date() -> None:
    """金额必须带牌价快照日期。

    没有日期的价格数字，三个月后就是一段**没法验证的历史**：
    你没法判断它是"当时的真实价格"还是"当时写错了"。
    """
    c = estimate_cost(model="deepseek-flash", prompt_tokens=1000)
    assert c.as_of != "unknown", "牌价表没读到"
    assert c.as_of in c.basis, "basis 里要带上日期，光存在字段里会被报表丢掉"


# --------------------------------------------------------------------------- #
# `billable`：这一轮的钱到底花没花
# --------------------------------------------------------------------------- #


def test_default_is_billable() -> None:
    """默认当成真花了钱。**这是保守的方向**：把没花的算成花了，
    最多是报表偏贵（会被追问）；把花了的算成没花，是漏账。"""
    assert estimate_cost(model="deepseek-flash", completion_tokens=10).billable is True


def test_is_free_and_billable_answer_different_questions() -> None:
    """`is_free` 问"发生没发生"，`billable` 问"计没计费"。**两者都要看。**

    重放画像是：`billable=False`（没花钱）但 `is_free=False`（token 不是 0，
    因为旧记录里记着 17255 个）。只看其中一个字段就会漏掉它。
    """
    c = estimate_cost(model="deepseek-flash", prompt_tokens=14352, completion_tokens=2903)
    c.billable = False  # 模拟 replay：数是真的，钱不是这次花的
    assert c.billable is False
    assert c.is_free is False, "token 不是 0，所以'什么都没发生'也为假"
    assert c.total_cny > 0, "它仍然记得'当初真调会花多少'——这个数做对照实验要用"

    nothing = estimate_cost(model="deepseek-flash")
    assert nothing.is_free is True and nothing.total_cny == 0.0


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def test_sum_adds_the_parts() -> None:
    a = estimate_cost(model="deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0)
    b = estimate_cost(model="deepseek-flash", prompt_tokens=0, completion_tokens=1_000_000)
    total = sum_costs([a, b])
    assert total.prompt_tokens == 1_000_000
    assert total.completion_tokens == 1_000_000
    assert total.total_cny == round(a.total_cny + b.total_cny, 8)


def test_sum_keeps_as_of_and_basis() -> None:
    """汇总**必须保留出处**。自己写个 `sum(...)` 的话这两个字段就丢了，
    而丢掉的正是"这个数字怎么来的"。"""
    a = estimate_cost(model="deepseek-flash", prompt_tokens=1000)
    total = sum_costs([a])
    assert total.as_of == a.as_of
    assert total.basis, "汇总没有出处，等于一个没法追问的数字"


def test_sum_propagates_billable() -> None:
    """**汇总里的 `billable` 必须显式传，不能让它取默认值 `True`。**

    这是本文件里最接近真机事故的一条：离线（`replay`）跑一遍演示，
    每次问答都标了"本次 0 元"，但汇总表上却出现 `¥0.266888`——
    因为 `sum_costs` 把每一笔的 `total_cny`（那是"若真调会花多少"）加起来，
    而 `billable` 用了默认的 `True`。
    """
    replayed = estimate_cost(model="deepseek-flash", prompt_tokens=1000, completion_tokens=100)
    replayed.billable = False
    assert sum_costs([replayed, replayed]).billable is False, (
        "整份汇总都是重放来的，却标成了'花过钱'"
    )

    real = estimate_cost(model="deepseek-flash", prompt_tokens=1000, completion_tokens=100)
    # 混着来的时候保守当成"有真金白银"——把花掉的钱说成没花，比反过来严重。
    assert sum_costs([replayed, real]).billable is True


def test_sum_of_nothing_is_empty_not_free() -> None:
    empty = sum_costs([])
    assert empty.basis == "无调用"
    assert empty.billable is False, "一次调用都没有，不该标成'花过钱'"


# --------------------------------------------------------------------------- #
# 人话渲染：三种"0 元"必须长得不一样
# --------------------------------------------------------------------------- #


def test_describe_cost_separates_the_four_kinds_of_zero() -> None:
    """**四种 `0 元` 必须说出四种不同的话。**

    1. 牌价表里缺这个模型 → "未记账"（这是个缺口，得去补表）
    2. 压根没调模型（拒答/闲聊） → "没有调用模型"（这是真的 0）
    3. 调了但一个 token 都没产出 → "没有产生 token"（也是真的 0）
    4. 读了旧记录（replay/缓存） → "本次 0 元（当初真调约 ¥X）"

    第 4 种最要命：它的**金额不是 0**（0.138），所以任何
    "金额大于 0 就显示金额"的写法都会把它报成一次真实消费。

    第 1 和第 3 种也不能混：混了的话，第 3 种会被当成"配置缺失"去排查，
    而真正缺配置的第 1 种就藏在那句"已经查过了"里，再没人看。
    """
    unpriced = estimate_cost(model="没配价的模型", prompt_tokens=100, completion_tokens=10)
    assert "未记账" in describe_cost(unpriced)

    # 拒答/闲聊走的是 API 里那条"没计过费的调用"的退路：**显式标 False**。
    # 这正是真实形状——不是 `estimate_cost()` 不带 token 就能表达出来的。
    refused = estimate_cost(model="deepseek-flash")
    refused.billable = False
    assert "没有调用模型" in describe_cost(refused)

    empty_call = estimate_cost(model="deepseek-flash")  # billable 仍是默认的 True
    assert "没有产生 token" in describe_cost(empty_call)

    replayed = estimate_cost(model="deepseek-flash", prompt_tokens=14352, completion_tokens=2903)
    replayed.billable = False
    text = describe_cost(replayed)
    assert "本次 0 元" in text, f"重放被报成了消费：{text!r}"
    assert "17255" in text, "重放的 token 数还是要显示的（那是它的来处）"
    assert replayed.total_cny > 0 and "¥" in text, "对照用的'若真调'金额不能丢"


def test_describe_cost_of_a_real_call_shows_the_breakdown() -> None:
    c = estimate_cost(
        model="deepseek-flash",
        prompt_tokens=14352,
        cache_hit_tokens=5504,
        cache_miss_tokens=8848,
        completion_tokens=2903,
        reasoning_tokens=1533,
    )
    text = describe_cost(c)
    assert "17255" in text, "总数自己加：入 + 出（CostBreakdown 里没有 total_tokens 字段）"
    assert "入 14352" in text and "出 2903" in text
    assert "思考 1533" in text
    assert "缓存命中 5504/14352" in text, "缓存命中率是省钱的直接证据，要显示"


@pytest.mark.parametrize("cost", [None])
def test_describe_cost_tolerates_none(cost) -> None:
    """没有成本信息时**明说没有**，不要显示成 0 元。

    `0 元` 是一个结论，`不知道` 是另一个。混起来就等于用"免费"
    掩盖了一个坏掉的记账环节。
    """
    assert "没有成本信息" in describe_cost(cost)


def test_cost_breakdown_is_still_serializable() -> None:
    """`billable` 必须能被 `model_dump()` 带出去——接口要靠它传给前端。

    如果它是 property 而不是字段，`model_dump()` 里就没有它，
    `CostOut` 那边拿到的是默认的 `True`，**前端会继续把重放报成消费**。
    """
    c = estimate_cost(model="deepseek-flash", prompt_tokens=10)
    c.billable = False
    assert c.model_dump()["billable"] is False
