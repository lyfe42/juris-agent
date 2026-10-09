"""F23 检索指标的测试。

三条最重要的断言：
1. **两种 Recall 在同一份结果上必须给出不同的数**——只报一种就是自欺。
2. **MRR 未命中记 0，不跳过**——跳过困难题会让指标虚高。
3. **缺排名的题按零召回计入**，不能从分母里消失。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from juris_agent.eval.retrieval_eval import (
    GoldItem,
    evaluate,
    expand_ranking,
    gold_hash,
    load_gold,
    score_one,
)
from juris_agent.ingest.chunkers import Chunk


def chunk(cid: str, keys: list[str]) -> Chunk:
    return Chunk(chunk_id=cid, text="t", article_keys=keys, primary_key=keys[0], law_id=keys[0].split("#")[0])


class TestTwoRecalls:
    """双口径是这一层的核心。gold 3 条只召回 1 条时，两个口径差 3 倍。"""

    def test_hit_and_art_differ_on_partial_recall(self) -> None:
        s = score_one(["a#1", "a#2", "a#3"], ["a#1", "x#9", "y#9"])
        assert s["recall_hit@5"] == 1.0  # 找到门了
        assert s["recall_art@5"] == pytest.approx(1 / 3)  # 但只覆盖三分之一

    def test_both_full_when_all_found(self) -> None:
        s = score_one(["a#1", "a#2"], ["a#1", "a#2"])
        assert s["recall_hit@5"] == 1.0
        assert s["recall_art@5"] == 1.0

    def test_both_zero_when_missed(self) -> None:
        s = score_one(["a#1"], ["z#1"])
        assert s["recall_hit@5"] == 0.0
        assert s["recall_art@5"] == 0.0

    def test_hit_is_k_cut_off(self) -> None:
        # gold 在第 6 位：Recall@1/5 为 0，Recall@10 为 1
        ranked = [f"x#{i}" for i in range(1, 6)] + ["a#1"]
        s = score_one(["a#1"], ranked)
        assert s["recall_hit@1"] == 0.0
        assert s["recall_hit@5"] == 0.0
        assert s["recall_hit@10"] == 1.0


class TestMRR:
    def test_reciprocal_rank(self) -> None:
        assert score_one(["a#1"], ["x#1", "a#1"])["mrr@10"] == pytest.approx(0.5)

    def test_miss_counts_as_zero_not_skipped(self) -> None:
        """未命中必须记 0。若"跳过"，困难题的 0 分会被排除在均值之外。"""
        assert score_one(["a#1"], ["x#1", "y#1"])["mrr@10"] == 0.0

    def test_beyond_cutoff_is_zero(self) -> None:
        ranked = [f"x#{i}" for i in range(1, 11)] + ["a#1"]
        assert score_one(["a#1"], ranked)["mrr@10"] == 0.0


class TestNDCG:
    def test_perfect_ranking_scores_one(self) -> None:
        assert score_one(["a#1"], ["a#1", "x#1"])["ndcg@5"] == pytest.approx(1.0)

    def test_lower_position_scores_less(self) -> None:
        first = score_one(["a#1"], ["a#1"])["ndcg@5"]
        second = score_one(["a#1"], ["x#1", "a#1"])["ndcg@5"]
        assert first > second

    def test_no_gold_is_zero(self) -> None:
        assert score_one([], ["a#1"])["ndcg@5"] == 0.0


class TestExpandRanking:
    def test_dedupes_keeping_first_rank(self) -> None:
        chunks = [chunk("c1", ["a#1", "a#2"]), chunk("c2", ["a#2", "a#3"])]
        assert expand_ranking(chunks) == ["a#1", "a#2", "a#3"]

    def test_respects_max_keys(self) -> None:
        chunks = [chunk("c1", ["a#1", "a#2"]), chunk("c2", ["a#3"])]
        assert expand_ranking(chunks, max_keys=2) == ["a#1", "a#2"]

    def test_multi_key_chunk_contributes_several(self) -> None:
        """定长切分的块覆盖多条——展开后一条 chunk 给出多个条级名次。"""
        assert len(expand_ranking([chunk("fw0", ["a#1", "a#2", "a#3"])])) == 3


class TestEvaluate:
    def g(self, id_: str, gold: list[str], **kw) -> GoldItem:
        return GoldItem(id=id_, query="q", gold=gold, **kw)

    def test_missing_ranking_counts_as_zero(self) -> None:
        """题在 gold 里、却没有任何排名结果 → 按零召回计入，**不从分母消失**。"""
        rep = evaluate([self.g("q1", ["a#1"])], rankings={})
        assert rep.overall.n == 1
        assert rep.overall.values["recall_hit@5"] == 0.0

    def test_groups_by_difficulty_and_qtype(self) -> None:
        gold = [
            self.g("q1", ["a#1"], difficulty="easy", qtype="single_fact"),
            self.g("q2", ["a#1", "a#2"], difficulty="hard", qtype="multi_article"),
        ]
        rep = evaluate(gold, {"q1": ["a#1"], "q2": ["a#1", "a#2"]})
        assert set(rep.by_difficulty) == {"easy", "hard"}
        assert set(rep.by_qtype) == {"single_fact", "multi_article"}
        assert rep.by_difficulty["easy"].values["recall_hit@5"] == 1.0

    def test_source_split_is_reported(self) -> None:
        """machine 造题措辞贴原文、指标偏高，所以 human / machine 必须分开报。"""
        gold = [self.g("q1", ["a#1"], source="human"), self.g("q2", ["a#1"], source="machine")]
        rep = evaluate(gold, {"q1": ["a#1"], "q2": ["z#1"]})
        assert rep.by_source["human"].values["recall_hit@5"] == 1.0
        assert rep.by_source["machine"].values["recall_hit@5"] == 0.0

    def test_markdown_has_meta_and_rows(self) -> None:
        rep = evaluate([self.g("q1", ["a#1"])], {"q1": ["a#1"]}, meta={"corpus_hash": "abc"})
        md = rep.to_markdown("t")
        assert "corpus_hash=abc" in md
        assert "全部" in md


class TestGoldIO:
    def test_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "g.jsonl"
        p.write_text(
            json.dumps({"id": "q1", "query": "问", "gold": ["a#1"]}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        items = load_gold(p)
        assert items[0].id == "q1" and items[0].gold == ["a#1"]

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        p = tmp_path / "g.jsonl"
        line = json.dumps({"id": "q1", "query": "问", "gold": ["a#1"]}, ensure_ascii=False)
        p.write_text(line + "\n" + line + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_gold(p)

    def test_hash_is_order_insensitive_but_content_sensitive(self) -> None:
        a = [GoldItem(id="q1", query="x", gold=["a#1"]), GoldItem(id="q2", query="y", gold=["a#2"])]
        assert gold_hash(a) == gold_hash(list(reversed(a)))
        b = [GoldItem(id="q1", query="x", gold=["a#9"])]
        assert gold_hash(a) != gold_hash(b)
