"""F6 分词与 BM25 的测试。

重点是**分词器实例隔离**——这条测试来自一次真实翻车：第一版用全局
`jieba.load_userdict()` 加载领域词典，消融矩阵在同一个进程里跑多臂，先加载的那臂
把词典泄漏给了后面所有臂，于是 `dict=on` 和 `dict=off` 的数字一模一样，
差点得出"领域词典没用"的结论。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from juris_agent.ingest.chunkers import Chunk
from juris_agent.retrieval.bm25 import BM25Retriever, Tokenizer

# 一个 jieba 默认一定会切散的长词：火箭 / 推进剂
LONG_WORD = "火箭推进剂"
FAKE_DICT = f"{LONG_WORD} 5 n\n"


@pytest.fixture
def dict_file(tmp_path: Path) -> Path:
    p = tmp_path / "userdict.txt"
    p.write_text(FAKE_DICT, encoding="utf-8")
    return p


class TestTokenizerIsolation:
    """词典不能跨实例泄漏。这条守的是消融矩阵的可信度。"""

    def test_dict_merges_term(self, dict_file: Path) -> None:
        tok = Tokenizer(dict_path=dict_file, use_dict=True)
        assert LONG_WORD in tok(f"关于{LONG_WORD}的规定")

    def test_no_dict_keeps_it_split(self, dict_file: Path) -> None:
        tok = Tokenizer(dict_path=dict_file, use_dict=False)
        assert LONG_WORD not in tok(f"关于{LONG_WORD}的规定")

    def test_no_leak_across_instances(self, dict_file: Path) -> None:
        """先建带词典的、再建不带的——顺序不能影响结果（全局状态就会影响）。"""
        with_dict = Tokenizer(dict_path=dict_file, use_dict=True)
        without = Tokenizer(dict_path=dict_file, use_dict=False)
        assert LONG_WORD in with_dict(LONG_WORD)
        assert LONG_WORD not in without(LONG_WORD)

    def test_missing_dict_file_is_not_fatal(self, tmp_path: Path) -> None:
        tok = Tokenizer(dict_path=tmp_path / "不存在.txt", use_dict=True)
        assert tok.use_dict is False
        assert tok("试用期")  # 仍然能分词


class TestTokenizerFiltering:
    def test_drops_stopwords_and_punctuation(self, dict_file: Path) -> None:
        tok = Tokenizer(dict_path=dict_file, use_dict=False)
        toks = tok("劳动合同的试用期，最长多久？")
        assert "的" not in toks  # 停用词
        assert "，" not in toks and "？" not in toks  # 标点
        assert "试用期" in toks or "试用" in toks

    def test_stopwords_exclude_legally_meaningful_terms(self, dict_file: Path) -> None:
        """「不」「应当」「可以」是实义词（不得解除 ≠ 可以解除），不能当停用词丢掉。

        直接钉停用词表本身，而不是钉 jieba 的分词结果——后者的切法会随词典变化，
        让测试变成在测 jieba 而不是测我们的决定。
        """
        tok = Tokenizer(dict_path=dict_file, use_dict=False)
        for word in ("不", "应当", "可以", "不得", "解除"):
            assert word not in tok.stopwords


def make_retriever(tokenizer: Tokenizer) -> BM25Retriever:
    chunks = [
        Chunk(chunk_id="a#1", text="经济补偿按劳动者在本单位工作的年限计算。",
              article_keys=["a#1"], primary_key="a#1", law_id="a"),
        Chunk(chunk_id="a#2", text="竞业限制的期限不得超过二年。",
              article_keys=["a#2"], primary_key="a#2", law_id="a"),
        Chunk(chunk_id="a#3", text="劳动争议申请仲裁的时效期间为一年。",
              article_keys=["a#3"], primary_key="a#3", law_id="a"),
    ]
    return BM25Retriever(chunks, tokenizer)


class TestBM25Retriever:
    def test_ranks_relevant_chunk_first(self, dict_file: Path) -> None:
        r = make_retriever(Tokenizer(dict_path=dict_file, use_dict=False))
        hits = r.search("竞业限制期限几年", top_k=3)
        assert hits[0][0].chunk_id == "a#2"

    def test_scores_are_descending(self, dict_file: Path) -> None:
        r = make_retriever(Tokenizer(dict_path=dict_file, use_dict=False))
        scores = [s for _, s in r.search("经济补偿怎么算", top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_no_overlap_returns_empty(self, dict_file: Path) -> None:
        """一个词都没中时不返回零分结果——否则等于给下游塞噪声。"""
        r = make_retriever(Tokenizer(dict_path=dict_file, use_dict=False))
        assert r.search("宇宙飞船推进器原理", top_k=3) == []

    def test_respects_top_k(self, dict_file: Path) -> None:
        r = make_retriever(Tokenizer(dict_path=dict_file, use_dict=False))
        assert len(r.search("劳动合同经济补偿仲裁", top_k=2)) <= 2

    def test_empty_query(self, dict_file: Path) -> None:
        r = make_retriever(Tokenizer(dict_path=dict_file, use_dict=False))
        assert r.search("", top_k=3) == []

    def test_empty_corpus_rejected(self, dict_file: Path) -> None:
        with pytest.raises(ValueError):
            BM25Retriever([], Tokenizer(dict_path=dict_file, use_dict=False))
