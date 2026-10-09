"""F8 精排的测试。**重点全在降级路径上。**

"精排提升 X 点"这种测试随便写写就行，真正值得钉的是：
精排慢/坏/超时的时候，**系统还能返回结果**。
没有降级，一次 GPU 抖动就能让用户看到 500——这比平均质量重要得多。

测试里用**假打分器**（sleep 或抛错）来制造超时，不去真的把模型跑超时：
后者不稳定、CI 跑不了、而且会让人不想跑测试。
"""

from __future__ import annotations

import time

import pytest

from juris_agent.ingest.chunkers import Chunk
from juris_agent.retrieval.rerank import RerankResult, Reranker, make_noop_scorer


def chunk(cid: str, text: str = "正文") -> Chunk:
    return Chunk(chunk_id=cid, text=text, article_keys=[cid], primary_key=cid,
                 law_id=cid.split("#")[0], char_count=len(text))


class _SlowScorer:
    """慢打分器：睡够时间就必定超时。"""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.called = False

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        self.called = True
        time.sleep(self.seconds)
        return [1.0] * len(pairs)


class _FastScorer:
    """按 (query, passage) 里是否含 "命中" 给分——用来验证排序真的被改了。"""

    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        return [10.0 if "命中" in p else 0.0 for _, p in pairs]


class _BoomScorer:
    def predict(self, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        raise RuntimeError("模型加载失败")


class TestHappyPath:
    def test_reorders_by_score(self) -> None:
        cands = [chunk("a#1", "无关"), chunk("a#2", "命中"), chunk("a#3", "无关2")]
        r = Reranker(_FastScorer()).rerank("q", cands)
        assert not r.degraded
        assert r.chunks[0].chunk_id == "a#2"
        assert r.scores[0] == 10.0

    def test_respects_top_k(self) -> None:
        cands = [chunk(f"a#{i}") for i in range(1, 6)]
        r = Reranker(_FastScorer()).rerank("q", cands, top_k=2)
        assert len(r.chunks) == 2

    def test_records_candidate_count(self) -> None:
        cands = [chunk(f"a#{i}") for i in range(1, 4)]
        assert Reranker(_FastScorer()).rerank("q", cands).n_candidates == 3

    def test_noop_scorer_keeps_input_order(self) -> None:
        cands = [chunk("a#1"), chunk("a#2"), chunk("a#3")]
        r = Reranker(type("S", (), {"predict": staticmethod(make_noop_scorer())})()).rerank("q", cands)
        assert [c.chunk_id for c in r.chunks] == ["a#1", "a#2", "a#3"]


class TestDegradation:
    """降级路径——这个模块存在的理由。"""

    def test_timeout_returns_fallback_not_raise(self) -> None:
        cands = [chunk(f"a#{i}") for i in range(1, 6)]
        r = Reranker(_SlowScorer(0.5), timeout_ms=20).rerank("q", cands, top_k=3)
        assert r.degraded is True
        assert r.warning is not None
        assert "超时" in r.warning
        # 关键断言：**结果照样返回**，而且是融合层的原始顺序
        assert [c.chunk_id for c in r.chunks] == ["a#1", "a#2", "a#3"]

    def test_timeout_elapsed_is_close_to_budget(self) -> None:
        """不能等打分器跑完才返回——那降级就没有意义了。"""
        cands = [chunk(f"a#{i}") for i in range(1, 4)]
        t0 = time.perf_counter()
        Reranker(_SlowScorer(2.0), timeout_ms=50).rerank("q", cands)
        assert time.perf_counter() - t0 < 1.0

    def test_scorer_exception_degrades(self) -> None:
        """打分器本身炸了也要降级——精排是增强不是必需。"""
        cands = [chunk(f"a#{i}") for i in range(1, 4)]
        r = Reranker(_BoomScorer()).rerank("q", cands)
        assert r.degraded is True
        assert "失败" in (r.warning or "")
        assert len(r.chunks) == 3

    def test_warning_does_not_claim_cancellation(self) -> None:
        """**文档诚实性断言。** Python 杀不掉线程，超时省的是用户等待不是算力。

        如果哪天有人把 warning 改写成"已取消精排"，这条会失败——
        那是句假话，面试官顺着问一句就露馅。
        """
        cands = [chunk("a#1")]
        r = Reranker(_SlowScorer(0.3), timeout_ms=10).rerank("q", cands)
        assert "取消" not in (r.warning or "")
        assert "仍在后台" in (r.warning or "")


class TestEdgeCases:
    def test_empty_candidates(self) -> None:
        r = Reranker(_FastScorer()).rerank("q", [])
        assert r.chunks == [] and r.degraded is False

    def test_blank_query_skips_model(self) -> None:
        """空 query 没必要跑模型——直接截断返回，不打分也不降级。"""
        cands = [chunk(f"a#{i}") for i in range(1, 4)]
        r = Reranker(_BoomScorer()).rerank("   ", cands, top_k=2)
        assert r.degraded is False
        assert len(r.chunks) == 2

    def test_result_model_roundtrip(self) -> None:
        r = RerankResult(chunks=[chunk("a#1")], scores=[1.0], n_candidates=1)
        assert RerankResult(**r.model_dump()).chunks[0].chunk_id == "a#1"
