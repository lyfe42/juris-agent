"""F5 稠密检索的测试。

**全部用假编码器**（直接给向量），不加载真模型——理由是速度与稳定：
真模型一次前向几百毫秒、首次还要下 100MB，把它塞进单元测试会让
`pytest` 从 5 秒变成几分钟，然后大家就不跑了。

真正要钉的不是"模型好不好用"，而是这层自己的逻辑：
1. 归一化后的点积排序正确；
2. **缓存必须与 chunk 列表逐条对齐**——顺序错位是最阴险的 bug（见下）；
3. `IVF` 后端的近似损失是**真实存在**的（否则"精确 vs 近似"那条消融轴没意义）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from juris_agent.ingest.chunkers import Chunk
from juris_agent.retrieval.dense import (
    BACKENDS,
    EmbeddingCache,
    FaissIndex,
    NumpyIndex,
)


def ck(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, text=f"文本-{cid}", article_keys=[cid],
                 primary_key=cid, law_id=cid.split("#")[0])


class _FakeEncoder:
    """把每个 chunk 映射成一个固定向量：`chunk_id` 的哈希决定方向。

    这样"查询 A 应该召回 A"是可断言的，而又完全不碰 torch。
    """

    model_name = "fake/model"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.document_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_documents(texts)


class TestNumpyIndex:
    def test_finds_nearest(self) -> None:
        idx = NumpyIndex()
        m = np.eye(4, dtype=np.float32)
        idx.add(m)
        # 查询 = 第 2 个单位向量 → 应该召回 index 2
        hits = idx.search(np.array([0, 1, 0, 0], dtype=np.float32), top_k=1)
        assert hits == [(1, pytest.approx(1.0))]

    def test_sorted_descending(self) -> None:
        idx = NumpyIndex()
        m = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
        idx.add(m)
        scores = [s for _, s in idx.search(np.array([1, 0], dtype=np.float32), top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_corpus(self) -> None:
        idx = NumpyIndex()
        idx.add(np.eye(2, dtype=np.float32))
        assert len(idx.search(np.array([1, 0], dtype=np.float32), top_k=99)) == 2

    def test_top_k_zero(self) -> None:
        idx = NumpyIndex()
        idx.add(np.eye(2, dtype=np.float32))
        assert idx.search(np.array([1, 0], dtype=np.float32), top_k=0) == []

    def test_is_exact(self) -> None:
        """精确后端必须真的精确：暴力算一遍对照。"""
        rng = np.random.default_rng(0)
        m = rng.standard_normal((50, 16)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        idx = NumpyIndex()
        idx.add(m)
        got = [i for i, _ in idx.search(q, top_k=5)]
        expected = list(np.argsort(-(m @ q))[:5])
        assert got == [int(i) for i in expected]


class TestFaissIndex:
    """FAISS 是可选依赖，没装就跳过——不让 `--extra ml` 缺失变成测试失败。"""

    def test_approximate_loses_recall(self) -> None:
        faiss = pytest.importorskip("faiss")
        rng = np.random.default_rng(1)
        n, d = 2000, 32
        m = rng.standard_normal((n, d)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        q = rng.standard_normal(d).astype(np.float32)
        q /= np.linalg.norm(q)

        exact = NumpyIndex()
        exact.add(m)
        exact_ids = {i for i, _ in exact.search(q, top_k=10)}

        approx = FaissIndex(nlist=64, nprobe=1)
        approx.add(m)
        approx_ids = {i for i, _ in approx.search(q, top_k=10)}

        # 这条断言是 F24「精确 vs IVF」那一轴的**存在理由**：
        # 如果近似检索的召回和精确一样，那条轴就没有任何信息量，应该删掉。
        assert approx_ids != exact_ids, "IVF(nprobe=1) 居然和精确一致，消融轴失去意义"

    def test_registered_in_backends(self) -> None:
        assert "faiss_ivf" in BACKENDS and "exact" in BACKENDS


class TestEmbeddingCache:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path)
        vecs = np.eye(3, dtype=np.float32)
        cache.save("m", "by_article", "abc123", vecs, ["a#1", "a#2", "a#3"])
        got = cache.load("m", "by_article", "abc123")
        assert got is not None
        loaded, ids = got
        assert np.allclose(loaded, vecs)
        assert ids == ["a#1", "a#2", "a#3"]

    def test_miss_on_different_corpus_hash(self, tmp_path: Path) -> None:
        """语料变了必须读不到旧向量。

        漏掉这一条，改了语料之后会读到过期向量，算出一组
        **看起来正常、实则错误**的历史指标——这类错最难查。
        """
        cache = EmbeddingCache(tmp_path)
        cache.save("m", "by_article", "old-hash", np.eye(2, dtype=np.float32), ["a#1", "a#2"])
        assert cache.load("m", "by_article", "new-hash") is None

    def test_miss_on_different_chunker(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(tmp_path)
        cache.save("m", "by_article", "h", np.eye(2, dtype=np.float32), ["a#1", "a#2"])
        assert cache.load("m", "fixed_512", "h") is None

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        assert EmbeddingCache(tmp_path).load("m", "c", "h") is None


class TestDenseRetriever:
    def _build(self, tmp_path: Path, **kw):
        from juris_agent.retrieval.dense import DenseRetriever

        chunks = [ck(f"a#{i}") for i in range(1, 6)]
        enc = _FakeEncoder()
        cache = EmbeddingCache(tmp_path)
        return DenseRetriever(chunks, enc, chunker_name="by_article",
                              corpus_hash="h1", cache=cache, **kw), enc

    def test_query_retrieves_itself(self, tmp_path: Path) -> None:
        """假编码器下，"文本-a#3" 的查询应当召回 a#3——闭环自检。"""
        r, _ = self._build(tmp_path)
        assert r.search("文本-a#3", top_k=1)[0][0].chunk_id == "a#3"

    def test_second_build_hits_cache(self, tmp_path: Path) -> None:
        from juris_agent.retrieval.dense import DenseRetriever

        r1, enc1 = self._build(tmp_path)
        assert enc1.document_calls == 1
        chunks = r1.chunks
        enc2 = _FakeEncoder()
        DenseRetriever(chunks, enc2, chunker_name="by_article", corpus_hash="h1",
                       cache=EmbeddingCache(tmp_path))
        assert enc2.document_calls == 0, "第二次构建没有走缓存，消融会重复编码"

    def test_cache_misalignment_is_rejected(self, tmp_path: Path) -> None:
        """**这条最要紧。** 缓存里的 chunk 顺序与当前列表不一致时必须丢弃重编码。

        顺序错位是静默的：向量数量对得上，检索能跑，结果看起来正常——
        但"第四十七条"的向量挂到了别的条上，召回的是错的条。
        比直接报错危险得多。
        """
        from juris_agent.retrieval.dense import DenseRetriever

        cache = EmbeddingCache(tmp_path)
        chunks = [ck(f"a#{i}") for i in range(1, 6)]
        # 故意存一份**顺序颠倒**的缓存，但数量相同
        wrong = _FakeEncoder().encode_documents([c.text for c in reversed(chunks)])
        cache.save("fake/model", "by_article", "h1", wrong, [c.chunk_id for c in reversed(chunks)])

        before = _FakeEncoder()
        DenseRetriever(chunks, before, chunker_name="by_article", corpus_hash="h1", cache=cache)
        assert before.document_calls == 1, "顺序不一致的缓存被误用了"

    def test_empty_query(self, tmp_path: Path) -> None:
        r, _ = self._build(tmp_path)
        assert r.search("   ", top_k=5) == []

    def test_rank_returns_chunk_ids(self, tmp_path: Path) -> None:
        r, _ = self._build(tmp_path)
        assert r.rank("文本-a#2", top_k=1) == ["a#2"]

    def test_rejects_empty_corpus(self, tmp_path: Path) -> None:
        from juris_agent.retrieval.dense import DenseRetriever

        with pytest.raises(ValueError):
            DenseRetriever([], _FakeEncoder(), cache=EmbeddingCache(tmp_path))

    def test_rejects_unknown_backend(self, tmp_path: Path) -> None:
        from juris_agent.retrieval.dense import DenseRetriever

        with pytest.raises(ValueError):
            DenseRetriever([ck("a#1")], _FakeEncoder(), backend="不存在")
