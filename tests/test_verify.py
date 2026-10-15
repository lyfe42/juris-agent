"""引用校验的测试（F25 的核心）。

## 这套测试的价值高于它看起来的样子

引用校验是整个项目里**唯一一个"回答质量"指标不需要模型参与**的地方。
它的正确性因此**完全落在这套测试上**——如果 `parse_citations` 漏认了一种写法，
报表会低估引用率；如果它多认了，报表会高估引用率（更危险，因为没人会去查）。
所以下面把"该认的"和"不该认的"**两边都写成测试**。
"""

from __future__ import annotations

import pytest

from juris_agent.agent.state import InjectedArticle
from juris_agent.agent.verify import (
    parse_citations,
    resolve_keys,
    strip_invalid,
    summarize,
    verify_citations,
)


def _inj(n: int) -> list[InjectedArticle]:
    return [
        InjectedArticle(
            index=i,
            article_key=f"lcl#{40 + i}",
            citation=f"《劳动合同法》第{40 + i}条",
            law="劳动合同法",
            article_no=f"第{40 + i}条",
            text=f"第 {i} 条的正文",
        )
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------- #
# 解析：该认的
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("根据第四十七条 [1]，应当支付。", [1]),
        ("依据 [1][2] 的规定。", [1, 2]),
        ("见 [1,2] 与 [3]。", [1, 2, 3]),
        ("见 [1，2]。", [1, 2]),          # 中文逗号
        ("见 [1、2]。", [1, 2]),          # 顿号
        ("见【1】。", [1]),               # 中文方括号
        ("见 [ 1 ] 。", [1]),             # 括号内有空格
        ("[10]", [10]),                   # 两位数
        ("[1] [1] [1]", [1]),             # 去重
    ],
)
def test_parse_recognises_expected_forms(text: str, expected: list[int]) -> None:
    assert parse_citations(text) == expected


def test_parse_preserves_first_seen_order() -> None:
    """**保序**（不是排序）：`[3][1]` 要抽成 `[3,1]`。

    前端按这个顺序列出处，和用户读答案时碰到引用的顺序一致。
    用 `set` 会丢掉这个信息，用 `sorted` 会把用户读的顺序打乱。
    """
    assert parse_citations("[3][1][3]") == [3, 1]


# --------------------------------------------------------------------------- #
# 解析：不该认的
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "2021年1月1日起施行",        # 年份（没括号，本来就不会中）
        "共 [2021] 元",              # 三位以上的数字不是编号
        "见 [第47条]",               # 里面不是纯数字
        "见 []",                     # 空
        "见 [a]",                    # 非数字
    ],
)
def test_parse_ignores_non_citations(text: str) -> None:
    assert parse_citations(text) == []


# --------------------------------------------------------------------------- #
# 报告：三个指标怎么算
# --------------------------------------------------------------------------- #


def test_report_splits_valid_and_invalid() -> None:
    r = verify_citations("见 [1] 和 [9]。", _inj(3))
    assert r.valid == [1]
    assert r.invalid == [9], "注入只有 3 条，[9] 是编的"
    assert r.cited == [1, 9]
    assert r.uncited == [2, 3]


def test_precision_penalises_fabricated_numbers() -> None:
    """**编造编号必须扣分。** 这是"引用有效率"存在的全部理由。"""
    r = verify_citations("见 [1] 和 [9]。", _inj(3))
    assert r.precision == pytest.approx(0.5)


def test_precision_is_one_when_no_fabrication() -> None:
    assert verify_citations("见 [1] 和 [2]。", _inj(3)).precision == 1.0


def test_precision_is_zero_when_nothing_cited() -> None:
    """⚠️ 没引用时定成 0 而不是 1。

    定成 1 的话，"什么都不引用"会拿到满分的有效率——
    **报表会被一个不引用任何依据的答案刷高**。定成 0 会让它拉低平均，
    逼你去看"为什么这条没引用"（答案在 `no_citation_rate` 里）。
    """
    assert verify_citations("法律有规定。", _inj(3)).precision == 0.0


def test_coverage_measures_how_much_of_the_context_was_used() -> None:
    r = verify_citations("见 [1]。", _inj(4))
    assert r.coverage == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# 删掉编造的编号
# --------------------------------------------------------------------------- #


def test_strip_removes_only_invalid_markers() -> None:
    out = strip_invalid("结论 [1]，另一条 [9]。", verify_citations("结论 [1]，另一条 [9]。", _inj(3)))
    assert "[1]" in out
    assert "[9]" not in out


def test_strip_keeps_the_sentence() -> None:
    """⚠️ **只删编号，不删句子。**

    整句删掉会让答案读起来像被打断。删掉编号之后，用户看到的是
    "一句没有出处的陈述"——**这是诚实的状态**：我们确实证明不了它。
    """
    out = strip_invalid("所以你可以要求赔偿 [9]。", verify_citations("x [9]", _inj(1)))
    assert "所以你可以要求赔偿" in out
    assert "[9]" not in out


def test_strip_handles_mixed_group() -> None:
    """`[1,9]` 里 1 有效、9 编造 → 只留 `[1]`。"""
    out = strip_invalid("见 [1,9]。", verify_citations("见 [1,9]。", _inj(3)))
    assert "1" in out and "9" not in out


def test_strip_is_noop_when_all_valid() -> None:
    text = "见 [1] 和 [2]。"
    assert strip_invalid(text, verify_citations(text, _inj(3))) == text


# --------------------------------------------------------------------------- #
# 键解析与聚合
# --------------------------------------------------------------------------- #


def test_resolve_keys_maps_index_to_article_key() -> None:
    r = verify_citations("见 [2]。", _inj(3))
    assert resolve_keys(r, _inj(3)) == ["lcl#42"]


def test_summarize_uses_macro_average() -> None:
    """**逐题平均，不是把所有引用加起来一起除。**

    两题：一题引对了（precision 1.0），一题引的全是编的（precision 0.0）。

    - macro（本题采用）：`(1.0 + 0.0) / 2 = 0.5`
    - micro（若把有效引用数除以总引用数）：`1 / 2 = 0.5`——这两题正好一样，
      所以要看出差别得用引用条数不等的例子。但**macro 是这里唯一说得通的口径**：
      每题等权。"引用了 20 条的题"不该压倒"引用了 1 条的题"。
    """
    reports = [
        verify_citations("见 [1]。", _inj(3)),  # p=1.0
        verify_citations("见 [9]。", _inj(3)),  # p=0.0（注入只有 3 条）
    ]
    s = summarize(reports)
    assert s["precision"] == pytest.approx(0.5)
    assert s["no_citation_rate"] == 0.0


def test_duplicate_fabricated_numbers_are_deduped() -> None:
    """⚠️ **同一条编造引用写 4 遍，和写 1 遍扣一样多的分。**

    这是刻意的语义：`cited` 是**去重后的编号集合**，"有多少条不同的引用是真的"
    才是有意义的度量；把同一个假号码重复 4 次并不比写 1 次更错。
    **代价**：一个疯狂重复 `[9]` 的答案不会被额外惩罚。
    这个取舍写在这里，免得以后有人看到 `[9][9][9][9]` 却只扣了 0.5 以为是 bug。
    """
    once = verify_citations("见 [1][9]。", _inj(3))
    many = verify_citations("见 [1][9][9][9][9][9]。", _inj(3))
    assert once.precision == many.precision == pytest.approx(0.5)


def test_summarize_counts_zero_citation_answers() -> None:
    reports = [verify_citations("没有引用。", _inj(3)), verify_citations("见 [1]。", _inj(3))]
    assert summarize(reports)["no_citation_rate"] == pytest.approx(0.5)


def test_summarize_handles_empty_input() -> None:
    assert summarize([])["n"] == 0.0
