"""F7 融合的测试。

核心不是"RRF 算得对"，而是**RRF 与加权求和的行为差异**——
因为简历和面试里要讲的是"我为什么选 RRF"，那必须是个可断言的性质，
不是一段说辞。

下面 `TestWeightedSumIsUnstable` / `TestRRFIsStable` 那一对，
就是把那句"加权求和的权重不可移植"写成可执行的断言。
"""

from __future__ import annotations

import pytest

from juris_agent.retrieval.fusion import (
    normalize_scores,
    rrf_fuse,
    weighted_sum_fuse,
)


class TestRRFBasics:
    def test_single_list_preserves_order(self) -> None:
        fused = rrf_fuse([["a", "b", "c"]])
        assert [d for d, _ in fused] == ["a", "b", "c"]

    def test_scores_descend(self) -> None:
        _, scores = zip(*rrf_fuse([["a", "b", "c"]]))
        assert list(scores) == sorted(scores, reverse=True)

    def test_doc_in_both_lists_beats_doc_in_one(self) -> None:
        """两路都召回 > 只有一路召回——这是融合最朴素的预期。"""
        fused = rrf_fuse([["a", "b"], ["a", "c"]])
        assert fused[0][0] == "a"

    def test_duplicate_id_counted_once(self) -> None:
        """同一 id 在一路里出现两次，只算首次名次。

        不加这条防线，某一路只要把同一个 id 刷进去就能抬自己的总分。
        """
        once = dict(rrf_fuse([["a", "b"]]))
        twice = dict(rrf_fuse([["a", "a", "b"]]))
        assert once["a"] == pytest.approx(twice["a"])

    def test_lower_k_amplifies_top_rank(self) -> None:
        """k 越小，头部名次的差距被放得越大。"""
        small = dict(rrf_fuse([["a", "b"]], k=1))
        large = dict(rrf_fuse([["a", "b"]], k=1000))
        assert small["a"] - small["b"] > large["a"] - large["b"]

    def test_k_can_flip_the_winner(self) -> None:
        """**k 不是无关紧要的常数**——这条是 E 组 k 扫描（F24）的存在理由。

        场景：`x` 只在第一路拿到第 1 名，`y` 在**两路都排第 10**。

            x 的分 = 1/(k+1)
            y 的分 = 2/(k+10)

        比大小：`2(k+1)` vs `k+10` → 分界在 `k = 8`。
        `k < 8` 时"独占第一"更值钱；`k > 8` 时"两路都榜上有名"更值钱。

        所以 k 的取值实际上是在**调一种偏好**：偏"冒尖"还是偏"共识"。
        拿不准就该扫一遍看数据，而不是抄一个 60。
        """
        rankings = [
            ["x"] + [f"p{i}" for i in range(1, 10)] + ["y"],
            ["z"] + [f"q{i}" for i in range(1, 10)] + ["y"],
        ]
        assert rrf_fuse(rankings, k=1)[0][0] == "x"
        assert rrf_fuse(rankings, k=30)[0][0] == "y"

    def test_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse([["a"]], k=0)

    def test_weights_length_must_match(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse([["a"], ["b"]], weights=[1.0])

    def test_weights_shift_ranking(self) -> None:
        a_only = [["a"], ["b"]]
        assert rrf_fuse(a_only, weights=[1.0, 1.0])[0][0] in {"a", "b"}
        assert rrf_fuse(a_only, weights=[10.0, 1.0])[0][0] == "a"

    def test_zero_weight_drops_that_ranking_entirely(self) -> None:
        """权重为 0 的那一路，**它独有的文档必须整条消失**。

        这是修过的一个真 bug：0 权重下 `w/(k+rank)` 正好等于 0，
        而 0 分文档照样被塞进结果——于是"权重设 0"变成"降权但不关"，
        G 组的自检行（`bm25=0` 必须等价于 dense-only）悄悄失效。

        这个断言必须同时检查"多出来的没了"和"剩下的排序没变"：
        只查前者的话，一个把所有分都清零的实现也能通过。
        """
        sparse = ["s1", "s2"]
        dense = ["d1", "d2"]
        zeroed = rrf_fuse([sparse, dense], weights=[0.0, 1.0])
        assert {d for d, _ in zeroed} == {"d1", "d2"}, "稀疏独有文档没被丢掉"
        assert [d for d, _ in zeroed] == [d for d, _ in rrf_fuse([dense])], "剩下的排序变了"
        assert all(s > 0 for _, s in zeroed), "结果里不该有 0 分的文档"

    def test_all_weights_zero_yields_nothing(self) -> None:
        """全 0 权重等价于"一路都没开"。返回空比返回一堆 0 分文档诚实。"""
        assert rrf_fuse([["a", "b"], ["c"]], weights=[0.0, 0.0]) == []


class TestRRFDeterminism:
    def test_tie_break_is_stable(self) -> None:
        """同分时按 id 字典序。**必须确定**——否则两次跑出的排名会漂移，
        消融表里的数字就无法复现。"""
        first = rrf_fuse([["a"], ["b"]])
        second = rrf_fuse([["b"], ["a"]])
        assert first == second

    def test_empty_input(self) -> None:
        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []


class TestRRFIsStable:
    """RRF 的性质：**在列表尾部追加不相关的文档，不影响已有文档的名次分**。"""

    def test_tail_addition_does_not_change_existing_scores(self) -> None:
        base = dict(rrf_fuse([["a", "b"], ["b"]]))
        with_tail = dict(rrf_fuse([["a", "b", "x", "y", "z"], ["b"]]))
        # a 仍是第 1、b 仍是第 2，x/y/z 排在它们后面，所以 a、b 分数一字不变
        assert base["a"] == pytest.approx(with_tail["a"])
        assert base["b"] == pytest.approx(with_tail["b"])

    def test_score_scale_of_a_ranking_never_enters(self) -> None:
        """RRF 的输入只有名次，没有分数——结构上就不可能被分数尺度影响。"""
        import inspect

        # 显式钉住签名：一旦有人给 rrf_fuse 加了 scores 参数，这条会失败，
        # 提醒他"尺度无关"这个卖点正在被破坏。
        assert "scores" not in inspect.signature(rrf_fuse).parameters


class TestWeightedSumIsUnstable:
    """加权求和的病：min-max 归一化让**一道题内所有文档互相牵制**。

    下面两条断言就是"权重不可移植"的具体形态。
    """

    def test_weak_champion_gets_full_credit(self) -> None:
        """一路全军覆没时，它的"冠军"照样被拉到 1.0，然后以满权重混进来。

        0.31 是个很差的 BM25 分（说明几乎没有字面重合），
        但 min-max 眼里"它是这一路的最高分"，于是给它满分。
        """
        normalized = normalize_scores([("垃圾", 0.31), ("更垃圾", 0.30), ("最垃圾", 0.29)])
        assert normalized["垃圾"] == pytest.approx(1.0)

    def test_adding_a_worse_doc_raises_an_unrelated_doc(self) -> None:
        """**这条是加权求和的致命处。**

        在 A 路尾部追加一个更差的文档 x，A 路的最小值被拉低，
        于是**名次没变、分数没变的 b** 归一化分从 0.0 涨到 0.5，
        加权总分跟着变——最终排名可能翻盘。

        b 自己什么都没变，是"同题里别的文档变了"影响到了它。
        这就是"同一组权重换一道题行为就变了"的机制。
        """
        before = dict(weighted_sum_fuse([[("a", 0.9), ("b", 0.5)], [("c", 1.0)]]))
        after = dict(weighted_sum_fuse([[("a", 0.9), ("b", 0.5), ("x", 0.1)], [("c", 1.0)]]))
        assert before["b"] == pytest.approx(0.0)
        assert after["b"] > 0.0
        assert after["b"] != pytest.approx(before["b"])

    def test_unretrieved_and_retrieved_last_are_indistinguishable(self) -> None:
        """在某一路没被召回 → 记 0；召回了但排最后 → 归一化后也是 0。

        两种**语义完全不同**的情况被压成同一个数，信息在这里丢了。
        """
        ranked = [("a", 0.9), ("b", 0.5)]
        # b 排最后 → 归一化 0；c 完全没出现 → 也是 0
        assert normalize_scores(ranked)["b"] == pytest.approx(0.0)


class TestNormalizeScores:
    def test_maps_to_unit_interval(self) -> None:
        n = normalize_scores([("a", 5.0), ("b", 3.0), ("c", 1.0)])
        assert n == {"a": 1.0, "b": 0.5, "c": 0.0}

    def test_all_equal_maps_to_one(self) -> None:
        """`max == min` 时原式是 0/0。取 1.0 = "这一路只有它，它代表这一路"，
        与 RRF 的行为一致（RRF 下它们并列第一）。"""
        assert normalize_scores([("a", 2.0), ("b", 2.0)]) == {"a": 1.0, "b": 1.0}

    def test_empty(self) -> None:
        assert normalize_scores([]) == {}


class TestWeightedSumBasics:
    def test_weights_reject_mismatch(self) -> None:
        with pytest.raises(ValueError):
            weighted_sum_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[1.0])

    def test_deterministic_tie_break(self) -> None:
        assert weighted_sum_fuse([[("a", 1.0)], [("b", 1.0)]]) == \
               weighted_sum_fuse([[("b", 1.0)], [("a", 1.0)]])
