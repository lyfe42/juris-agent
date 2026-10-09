"""F10 编排的测试。

钉三件事：
1. **每一段的耗时与候选数都被记下来**（否则"检索结果为空"没法定位）。
2. **去重是两级**：chunk_id 与 primary_key，漏一级会让同一个法条在上下文里出现两次。
3. **邻条扩展不影响原始命中的排序**——"顺带捞的"不能挤掉"真正命中的"。
"""

from __future__ import annotations

import pytest

from juris_agent.ingest.chunkers import Chunk
from juris_agent.ingest.clean import Article
from juris_agent.retrieval.pipeline import (
    ArmConfig,
    RetrievalPipeline,
    expand_neighbors,
)


def ck(cid: str, key: str | None = None, text: str = "正文") -> Chunk:
    key = key or cid
    return Chunk(chunk_id=cid, text=text, article_keys=[key], primary_key=key,
                 law_id=key.split("#")[0], char_count=len(text))


class _FakeRetriever:
    """按预设表返回结果。`table[query]` 不存在时返回空。"""

    def __init__(self, table: dict[str, list[tuple[Chunk, float]]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        self.calls.append(query)
        return self.table.get(query, [])[:top_k]


class _FakeRewriter:
    name = "fake"

    def __init__(self, variants: list[str]) -> None:
        self.variants = variants

    def rewrite(self, query: str) -> list[str]:
        return self.variants


class TestStageInstrumentation:
    def test_records_every_stage(self) -> None:
        r = _FakeRetriever({"q": [(ck("lcl#47"), 1.0)]})
        pipe = RetrievalPipeline(ArmConfig(), sparse=r)
        res = pipe.search("q")
        names = [s.stage for s in res.stages]
        assert names[0] == "rewrite"
        assert "bm25" in names
        assert "fuse/rrf" in names
        assert names[-1] == "dedup"

    def test_elapsed_is_nonnegative_and_summed(self) -> None:
        r = _FakeRetriever({"q": [(ck("lcl#47"), 1.0)]})
        res = RetrievalPipeline(ArmConfig(), sparse=r).search("q")
        assert all(s.elapsed_ms >= 0 for s in res.stages)
        assert res.elapsed_ms == pytest.approx(sum(s.elapsed_ms for s in res.stages))

    def test_candidate_counts_expose_empty_results(self) -> None:
        """结果为空时，`n_out=0` 是在检索段还是在去重段，一眼看出来。"""
        r = _FakeRetriever({})  # 什么都没有
        res = RetrievalPipeline(ArmConfig(), sparse=r).search("q")
        bm25 = next(s for s in res.stages if s.stage == "bm25")
        assert bm25.n_out == 0

    def test_multi_variant_query_is_noted(self) -> None:
        r = _FakeRetriever({"a": [(ck("lcl#47"), 1.0)], "b": [(ck("lcl#46"), 0.5)]})
        pipe = RetrievalPipeline(ArmConfig(), sparse=r, rewriter=_FakeRewriter(["a", "b"]))
        res = pipe.search("原始")
        bm25 = next(s for s in res.stages if s.stage == "bm25")
        assert bm25.n_in == 2
        assert "2 个查询变体" in bm25.note
        assert r.calls == ["a", "b"]


class TestDedup:
    def test_dedupes_duplicate_chunk_ids(self) -> None:
        """同一个变体与另一个变体命中同一块 → 只留一个。"""
        r = _FakeRetriever({"a": [(ck("lcl#47"), 1.0)], "b": [(ck("lcl#47"), 0.9)]})
        pipe = RetrievalPipeline(ArmConfig(), sparse=r, rewriter=_FakeRewriter(["a", "b"]))
        assert [c.chunk_id for c in pipe.search("q").chunks] == ["lcl#47"]

    def test_dedupes_by_primary_key(self) -> None:
        """按条切 + 邻条扩展会撞：同一**条**被两个不同 chunk 覆盖。

        只按 chunk_id 去重的话，第四十七条会在上下文里出现两次，
        既浪费 token 又让引用编号重复。
        """
        r = _FakeRetriever({"q": [(ck("lcl#47"), 1.0), (ck("lcl#47+nb", "lcl#47"), 0.8)]})
        res = RetrievalPipeline(ArmConfig(), sparse=r).search("q")
        assert [c.chunk_id for c in res.chunks] == ["lcl#47"]

    def test_dedup_stage_counts(self) -> None:
        r = _FakeRetriever({"q": [(ck("a#1"), 1.0), (ck("a#1x", "a#1"), 0.9), (ck("a#2"), 0.8)]})
        res = RetrievalPipeline(ArmConfig(), sparse=r).search("q")
        dedup = next(s for s in res.stages if s.stage == "dedup")
        assert dedup.n_in == 3 and dedup.n_out == 2


class TestFusionInPipeline:
    def test_single_path_skips_fusion_naming(self) -> None:
        r = _FakeRetriever({"q": [(ck("a#1"), 1.0)]})
        res = RetrievalPipeline(ArmConfig(use_bm25=True, use_dense=False), sparse=r).search("q")
        assert any(s.stage == "fuse/rrf" for s in res.stages)

    def test_two_paths_both_enter_fusion(self) -> None:
        s = _FakeRetriever({"q": [(ck("a#1"), 1.0)]})
        d = _FakeRetriever({"q": [(ck("a#2"), 0.9)]})
        res = RetrievalPipeline(ArmConfig(use_bm25=True, use_dense=True), sparse=s, dense=d).search("q")
        assert {c.chunk_id for c in res.chunks} == {"a#1", "a#2"}

    def test_doc_hit_by_both_paths_ranks_first(self) -> None:
        s = _FakeRetriever({"q": [(ck("a#1"), 1.0), (ck("a#2"), 0.9)]})
        d = _FakeRetriever({"q": [(ck("a#2"), 0.9), (ck("a#3"), 0.8)]})
        res = RetrievalPipeline(ArmConfig(use_bm25=True, use_dense=True), sparse=s, dense=d).search("q")
        assert res.chunks[0].chunk_id == "a#2"

    def test_weighted_sum_fusion_runs(self) -> None:
        s = _FakeRetriever({"q": [(ck("a#1"), 1.0)]})
        d = _FakeRetriever({"q": [(ck("a#2"), 0.9)]})
        cfg = ArmConfig(fusion="weighted_sum")
        res = RetrievalPipeline(cfg, sparse=s, dense=d).search("q")
        assert any(st.stage == "fuse/weighted_sum" for st in res.stages)

    def test_unknown_fusion_rejected(self) -> None:
        s = _FakeRetriever({"q": [(ck("a#1"), 1.0)]})
        d = _FakeRetriever({"q": [(ck("a#2"), 0.9)]})
        with pytest.raises(ValueError):
            RetrievalPipeline(ArmConfig(fusion="不存在"), sparse=s, dense=d).search("q")

    def test_rrf_k_is_taken_from_config_not_from_default(self) -> None:
        """k 必须来自 `ArmConfig`，**不能**用 `rrf_fuse` 的默认 60。

        这条如果漏了，E 组（k 扫描）会变成五条一模一样的臂——
        表照样出、数字照样有，但"k 该选多少"这个结论是假的。
        所以这里把管道的输出和"直接按 config.rrf_k 算"的结果对齐比较。
        """
        from juris_agent.retrieval.fusion import rrf_fuse

        # 构造一组 k 敏感的数据：x 独占第一路榜首，y 两路都排第 10
        s_ids = ["x"] + [f"p{i}" for i in range(1, 10)] + ["y"]
        d_ids = ["z"] + [f"q{i}" for i in range(1, 10)] + ["y"]
        s = _FakeRetriever({"q": [(ck(i), 1.0 - n * 0.01) for n, i in enumerate(s_ids)]})
        d = _FakeRetriever({"q": [(ck(i), 1.0 - n * 0.01) for n, i in enumerate(d_ids)]})

        for k, expected_top in ((1, "x"), (30, "y")):
            # fused_k 开大，否则管道会在 20 条处截断，比较就变成"截断 vs 全量"
            cfg = ArmConfig(rrf_k=k, fused_k=100, top_k=100)
            got = [c.chunk_id for c in RetrievalPipeline(cfg, sparse=s, dense=d).search("q").chunks]
            want = [i for i, _ in rrf_fuse([s_ids, d_ids], k=k)]
            assert got == want, f"k={k} 时管道没按配置的 k 融合"
            assert got[0] == expected_top

    def test_default_rrf_k_matches_module_default(self) -> None:
        from juris_agent.retrieval.fusion import DEFAULT_RRF_K

        assert ArmConfig().rrf_k == DEFAULT_RRF_K

    def test_zero_weight_turns_a_path_off(self) -> None:
        """`sparse_weight=0` 必须与"根本没有这一路"完全等价。

        这是 G 组（权重扫描）的**自检行**：如果 w=0 的结果与 dense-only 不一致，
        说明权重没有真的作用在排序上，整组扫描的数字都不可信。
        """
        s_ids = ["a#1", "a#2", "a#3"]
        d_ids = ["a#3", "a#4", "a#1"]
        s = _FakeRetriever({"q": [(ck(i), 1.0 - n * 0.1) for n, i in enumerate(s_ids)]})
        d = _FakeRetriever({"q": [(ck(i), 1.0 - n * 0.1) for n, i in enumerate(d_ids)]})

        zero = RetrievalPipeline(
            ArmConfig(sparse_weight=0.0, fused_k=50, top_k=50), sparse=s, dense=d
        ).search("q")
        dense_only = RetrievalPipeline(
            ArmConfig(use_bm25=False, use_dense=True, fused_k=50, top_k=50), sparse=None, dense=d
        ).search("q")
        assert [c.chunk_id for c in zero.chunks] == [c.chunk_id for c in dense_only.chunks]

    def test_weights_change_the_ranking(self) -> None:
        """减小 BM25 的权重，必须把**只有 BM25 召回的**条往下压。

        构造要点：两路的 id **完全不重叠**。如果让同一个 id 出现在两路里，
        它会因为"两路都投了票"而天然占据高位——那是 RRF 本来的行为，
        与权重无关，用了那种构造就测不出权重到底有没有生效（第一版就是这么错的：
        `a#1` 在稀疏排第 1、稠密排第 3，加权后总分 0.05/61 + 1/63 仍压过稠密的
        第一名，断言 `== "a#9"` 失败——**失败的是测试，不是代码**）。

        不重叠之后，每个 id 只有一路投票，权重就直接决定它排在哪：
        等权时 `s1` 与 `d1` 同为 `1/61`，按 id 字典序 `d1` 在前，`s1` 紧跟其后；
        把稀疏权重降到 0.05，`s1` 变成 `0.05/61 ≈ 0.0008`，掉到三条稠密命中之后。
        """
        s_ids = ["s1", "s2", "s3"]
        d_ids = ["d1", "d2", "d3"]
        s = _FakeRetriever({"q": [(ck(i), 1.0 - n * 0.1) for n, i in enumerate(s_ids)]})
        d = _FakeRetriever({"q": [(ck(i), 1.0 - n * 0.1) for n, i in enumerate(d_ids)]})

        equal = RetrievalPipeline(
            ArmConfig(sparse_weight=1.0, dense_weight=1.0, fused_k=50, top_k=50), sparse=s, dense=d
        ).search("q")
        dense_biased = RetrievalPipeline(
            ArmConfig(sparse_weight=0.05, dense_weight=1.0, fused_k=50, top_k=50), sparse=s, dense=d
        ).search("q")

        equal_ids = [c.chunk_id for c in equal.chunks]
        biased_ids = [c.chunk_id for c in dense_biased.chunks]
        assert equal_ids != biased_ids, "权重改了但排序没变——权重根本没接进融合"
        # 等权时稀疏的第一名排在第 2 位（与 d1 同分，按 id 字典序 d1 在前）
        assert equal_ids.index("s1") == 1
        # 降权后它掉到三条稠密命中之后
        assert biased_ids.index("s1") == 3
        # 稠密那三条的相对顺序不受影响：权重只缩放，不改写单路内部名次
        assert biased_ids[:3] == ["d1", "d2", "d3"]


class TestArticleKeyExpansion:
    def test_expands_multikey_chunk_in_order(self) -> None:
        c = Chunk(chunk_id="fw0", text="t", article_keys=["a#1", "a#2", "a#3"],
                  primary_key="a#1", law_id="a", char_count=1)
        res = RetrievalPipeline(ArmConfig(), sparse=_FakeRetriever({"q": [(c, 1.0)]})).search("q")
        assert res.expand_article_keys() == ["a#1", "a#2", "a#3"]

    def test_expansion_dedupes(self) -> None:
        c1 = Chunk(chunk_id="c1", text="t", article_keys=["a#1", "a#2"], primary_key="a#1",
                   law_id="a", char_count=1)
        c2 = Chunk(chunk_id="c2", text="t", article_keys=["a#2", "a#3"], primary_key="a#2",
                   law_id="a", char_count=1)
        res = RetrievalPipeline(ArmConfig(), sparse=_FakeRetriever({"q": [(c1, 1.0), (c2, 0.9)]})).search("q")
        assert res.expand_article_keys() == ["a#1", "a#2", "a#3"]


def art(key: str, num: int, text: str) -> Article:
    return Article(article_key=key, law_id=key.split("#")[0], law_name="测试法",
                   law_short="测试法", article_no=f"第{num}条", article_num=num,
                   chapter=None, text=text, char_count=len(text))


class TestNeighborExpansion:
    def test_adds_adjacent_articles_after_hits(self) -> None:
        articles = {f"a#{i}": art(f"a#{i}", i, f"第{i}条正文") for i in range(1, 6)}
        hits = [ck("a#3")]
        out = expand_neighbors(hits, articles, window=1)
        assert [c.primary_key for c in out] == ["a#3", "a#2", "a#4"]
        assert out[0].chunk_id == "a#3"  # 原始命中仍排第一

    def test_does_not_cross_law_boundary(self) -> None:
        articles = {"a#1": art("a#1", 1, "甲"), "b#1": art("b#1", 1, "乙")}
        out = expand_neighbors([ck("a#1")], articles, window=1)
        assert {c.law_id for c in out} == {"a"}

    def test_window_zero_is_noop(self) -> None:
        assert expand_neighbors([ck("a#1")], {}, window=0) == [ck("a#1")]

    def test_never_duplicates(self) -> None:
        articles = {f"a#{i}": art(f"a#{i}", i, f"第{i}条") for i in range(1, 6)}
        out = expand_neighbors([ck("a#3"), ck("a#2")], articles, window=2)
        keys = [c.primary_key for c in out]
        assert len(keys) == len(set(keys))

    def test_neighbor_chunks_carry_own_key_only(self) -> None:
        """扩展来的 chunk 的 `article_keys` 必须只有它自己——
        否则会污染条级排名口径，把没真正命中的条算成命中。"""
        articles = {f"a#{i}": art(f"a#{i}", i, f"第{i}条") for i in range(1, 4)}
        out = expand_neighbors([ck("a#2")], articles, window=1)
        for c in out[1:]:
            assert c.article_keys == [c.primary_key]


class TestConstructionValidation:
    def test_requires_at_least_one_retriever(self) -> None:
        with pytest.raises(ValueError):
            RetrievalPipeline(ArmConfig())

    def test_armconfig_describe_is_readable(self) -> None:
        cfg = ArmConfig(chunker="by_article", use_bm25=True, use_dense=True,
                        use_rerank=True, rewrite="synonym", neighbor_window=1)
        label = cfg.describe()
        for token in ("by_article", "bm25", "dense", "rerank", "rw:synonym", "nb1"):
            assert token in label
