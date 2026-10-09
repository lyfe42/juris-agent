"""F2 切分器的测试。

契约只有两条，但四条臂都必须满足：
1. **不漏条**：所有 chunk 覆盖的 `article_keys` 并起来 == 全语料。
2. **chunk_id 唯一**：它是检索结果的主键，重了就没法去重、没法回溯。
   `fixed_window` 的 id 由字符偏移生成，最容易在这里出错。
"""

from __future__ import annotations

import pytest

from juris_agent.ingest.chunkers import (
    CHUNKERS,
    by_article,
    by_article_plus_context,
    fixed_window,
    sentence_window,
)
from juris_agent.ingest.clean import Article


def art(key: str, num: int, text: str, chapter: str | None = "第一章 总则") -> Article:
    return Article(
        article_key=key,
        law_id=key.split("#")[0],
        law_name="测试法",
        law_short="测试法",
        article_no=f"第{num}条",
        article_num=num,
        chapter=chapter,
        text=text,
        char_count=len(text),
    )


@pytest.fixture
def articles() -> list[Article]:
    return [
        art("t#1", 1, "第一条正文。" * 3),
        art("t#2", 2, "本法适用于下列情形：（一）甲；（二）乙。"),
        art("t#3", 3, "分则第一条。", chapter="第二章 分则"),
        art("u#1", 1, "另一部法的第一条，内容更长一些。" * 4),
    ]


ALL_CHUNKERS = [
    by_article,
    by_article_plus_context,
    lambda a: fixed_window(a, size=64, overlap=16),
    lambda a: sentence_window(a, window=2, stride=1),
]


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda f: getattr(f, "__name__", "fixed64"))
def test_no_article_is_lost(articles, chunker) -> None:
    chunks = chunker(articles)
    covered = {k for c in chunks for k in c.article_keys}
    assert covered == {a.article_key for a in articles}


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda f: getattr(f, "__name__", "fixed64"))
def test_chunk_ids_unique(articles, chunker) -> None:
    ids = [c.chunk_id for c in chunker(articles)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda f: getattr(f, "__name__", "fixed64"))
def test_no_empty_chunk(articles, chunker) -> None:
    assert all(c.text.strip() for c in chunker(articles))


class TestByArticle:
    def test_one_chunk_per_article(self, articles) -> None:
        chunks = by_article(articles)
        assert [c.chunk_id for c in chunks] == ["t#1", "t#2", "t#3", "u#1"]
        assert all(len(c.article_keys) == 1 for c in chunks)

    def test_text_is_verbatim(self, articles) -> None:
        assert by_article(articles)[1].text == articles[1].text


class TestByArticlePlusContext:
    def test_prepends_chapter(self, articles) -> None:
        assert by_article_plus_context(articles)[0].text.startswith("【第一章 总则】")

    def test_context_follows_chapter_switch(self, articles) -> None:
        assert "第二章 分则" in by_article_plus_context(articles)[2].text

    def test_no_chapter_means_no_prefix(self) -> None:
        a = art("t#1", 1, "正文。", chapter=None)
        assert by_article_plus_context([a])[0].text == "正文。"


class TestFixedWindow:
    def test_can_span_multiple_articles(self) -> None:
        """定长窗口的**定义特征**就是不看条界：一块可能覆盖多条。"""
        arts = [art(f"t#{i}", i, "短条文。") for i in range(1, 21)]
        chunks = fixed_window(arts, size=60, overlap=0)
        assert any(len(c.article_keys) > 1 for c in chunks)

    def test_window_size_respected(self) -> None:
        arts = [art(f"t#{i}", i, "x" * 100) for i in range(1, 6)]
        assert all(c.char_count <= 50 for c in fixed_window(arts, size=50, overlap=10))

    def test_law_boundary_not_crossed(self) -> None:
        """不同法规不能在同一个窗口里——拼接跨法条会把无关文本缝在一起。"""
        arts = [art("t#1", 1, "a" * 40), art("u#1", 1, "b" * 40)]
        for c in fixed_window(arts, size=100, overlap=0):
            assert len({k.split("#")[0] for k in c.article_keys}) == 1

    def test_rejects_bad_overlap(self) -> None:
        with pytest.raises(ValueError):
            fixed_window([art("t#1", 1, "正文")], size=10, overlap=10)


class TestSentenceWindow:
    def test_keys_ordered_by_first_appearance(self) -> None:
        chunks = sentence_window([art("t#1", 1, "甲。乙。丙。")], window=3, stride=1)
        assert chunks[0].article_keys == ["t#1"]

    def test_stride_controls_chunk_count(self) -> None:
        arts = [art("t#1", 1, "一。二。三。四。五。六。")]
        s1 = len(sentence_window(arts, window=2, stride=1))
        s2 = len(sentence_window(arts, window=2, stride=2))
        assert s1 > s2

    def test_rejects_bad_params(self) -> None:
        with pytest.raises(ValueError):
            sentence_window([art("t#1", 1, "正文。")], window=0)


def test_registry_covers_all_arms() -> None:
    assert set(CHUNKERS) == {
        "fixed_512",
        "by_article",
        "sentence_window",
        "by_article_plus_context",
    }
