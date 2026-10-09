"""F1 清洗层的测试。

分两层：
- **单元**：中文数字解析、合成文本上的切分（目录 / 跨行 / 之Y / 行内引用）。这部分
  不依赖 data/raw，永远能跑，是 CI 的主力。
- **集成**：对真实语料跑一遍，断言每部法规的条数、条号连续、章节归属。
  语料不在（干净 clone）时自动跳过——它测的是"数据没坏"，不是"代码没坏"。
"""

from __future__ import annotations

import pytest

from juris_agent.ingest.clean import (
    Article,
    LawMeta,
    clean_corpus,
    cn_to_int,
    corpus_hash,
    load_manifest,
    parse_articles,
)
from juris_agent.config import get_settings

META = LawMeta(id="t", file="t.docx", name="测试法", short="测试法")


class TestCnToInt:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("一", 1),
            ("十", 10),
            ("十一", 11),
            ("十五", 15),
            ("二十", 20),
            ("二十三", 23),
            ("三十", 30),
            ("四十七", 47),
            ("九十九", 99),
            ("一百", 100),
            ("一百零五", 105),
            ("二百二十", 220),
            ("一千", 1000),
            ("一千零一", 1001),
            ("一千二百六十", 1260),  # 民法典末条
        ],
    )
    def test_parse(self, s: str, expected: int) -> None:
        assert cn_to_int(s) == expected

    def test_rejects_unknown_char(self) -> None:
        with pytest.raises(ValueError):
            cn_to_int("第一")


SYNTHETIC = """\
中华人民共和国测试法

（2020年1月1日通过）

目　　录
第一章　总　　则
第二章　分　　则

第一章　总　　则

第一条　为了测试，制定本法。
第二条　本法适用于下列情形：
（一）第一种情形；
（二）第二种情形。
依照本法第八条规定办理。
第三条　用语含义如下：
本法所称"以上"，包括本数。

第二章　分　　则

第四条　分则第一条。
第五条　分则第二条。
第六条　应当依照本法第二条、第三条处理。
"""


class TestParseArticles:
    """合成文本覆盖四个真实脏点：目录、跨行、行内引用、章节标题。"""

    def parse(self) -> list[Article]:
        return parse_articles(SYNTHETIC, META)

    def test_article_count_excludes_toc(self) -> None:
        # 目录里的"第二章"不是条文，第一条之前的标题行也不该被当成条文
        assert [a.article_num for a in self.parse()] == [1, 2, 3, 4, 5, 6]

    def test_multiline_body_merged(self) -> None:
        a2 = self.parse()[1]
        assert "（二）第二种情形" in a2.text
        assert "依照本法第八条规定办理" in a2.text  # 行内引用要保留在正文里
        assert a2.article_key == "t#2"

    def test_toc_not_mistaken_for_body(self) -> None:
        # 目录区的两行标题必须被丢掉：正文第一条属于「第一章 总则」
        assert self.parse()[0].chapter == "第一章 总则"

    def test_chapter_switches_at_heading(self) -> None:
        arts = self.parse()
        assert arts[2].chapter == "第一章 总则"  # 第三条
        assert arts[3].chapter == "第二章 分则"  # 第四条

    def test_heading_not_merged_into_previous_article(self) -> None:
        # 第三条后面紧跟"第二章"标题，标题不能落进第三条的正文
        assert "第二章" not in self.parse()[2].text

    def test_cross_reference_mid_line_does_not_split(self) -> None:
        # 第二条正文里出现"本法第八条"，不能因此切出新条
        assert len(self.parse()) == 6

    def test_article_key_is_stable_primary_key(self) -> None:
        keys = [a.article_key for a in self.parse()]
        assert keys[0] == "t#1" and keys[-1] == "t#6"

    def test_zhitu_key(self) -> None:
        text = "第一条　正文。\n第二条之一　补充规定。\n第三条　正文。"
        arts = parse_articles(text, META)
        assert [a.article_key for a in arts] == ["t#1", "t#2之一", "t#3"]

    def test_no_article_returns_empty(self) -> None:
        assert parse_articles("这里没有任何条文。", META) == []


class TestEffectiveDate:
    """生效日期的优先级：manifest > 正文正则 > 无。**顺序反了就出法律事故。**"""

    def test_manifest_wins_over_text(self) -> None:
        # 正文写 1995-01-01（原始版本），清单写 2018-12-29（现行修正版）
        text = "第一条　本法自1995年1月1日起施行。"
        meta = LawMeta(
            id="x", file="x.docx", name="X法", short="X法", effective_date="2018-12-29"
        )
        art = parse_articles(text, meta)[0]
        assert str(art.effective_date) == "2018-12-29"
        assert art.effective_date_source == "manifest"

    def test_falls_back_to_text_regex(self) -> None:
        text = "第一条　本法自1995年1月1日起施行。"
        art = parse_articles(text, META)[0]
        assert str(art.effective_date) == "1995-01-01"
        assert art.effective_date_source == "text-regex"

    def test_source_none_when_neither(self) -> None:
        art = parse_articles("第一条　正文。", META)[0]
        assert art.effective_date is None
        assert art.effective_date_source == "none"


# --------------------------------------------------------------------------- #
# 集成：对真实语料
# --------------------------------------------------------------------------- #

RAW = get_settings().paths.raw
MANIFEST = get_settings().paths.configs / "laws_manifest.yaml"

# 这 8 部法规的条数是**法定事实**，不是拟合出来的。它们对上，才说明切分没错。
EXPECTED_COUNTS = {
    "lcl": 98,
    "ldf": 107,
    "lcl_tiaoli": 38,
    "shbx": 98,
    "tjzc": 54,
    "mfd": 1260,
    "js1": 54,
    "js2": 21,
}


@pytest.fixture(scope="module")
def corpus() -> list[Article]:
    """整个真实语料清洗一遍（约 1 秒）。模块级共享，别每个用例重跑。"""
    articles, _ = clean_corpus(RAW, load_manifest(MANIFEST))
    return articles


@pytest.mark.skipif(not RAW.exists() or not any(RAW.glob("*.docx")), reason="语料不在，跳过集成测试")
class TestRealCorpus:
    def test_each_law_article_count(self, corpus: list[Article]) -> None:
        got: dict[str, int] = {}
        for a in corpus:
            got[a.law_id] = got.get(a.law_id, 0) + 1
        assert got == EXPECTED_COUNTS

    def test_article_numbers_are_contiguous(self, corpus: list[Article]) -> None:
        """条号必须是 1..N 无洞——丢一条就说明跨行合并或正则错了。"""
        by_law: dict[str, list[int]] = {}
        for a in corpus:
            by_law.setdefault(a.law_id, []).append(a.article_num)
        for law_id, nums in by_law.items():
            assert nums == list(range(1, len(nums) + 1)), f"{law_id} 条号不连续"

    def test_no_empty_body(self, corpus: list[Article]) -> None:
        empty = [a.article_key for a in corpus if not a.text.strip()]
        assert empty == []

    def test_chapter_attribution(self, corpus: list[Article]) -> None:
        # 只有两个司法解释没有章节体例；其余每一条都必须有章
        no_chapter = {a.law_id for a in corpus if not a.chapter}
        assert no_chapter == {"js1", "js2"}

    def test_key_articles_content(self, corpus: list[Article]) -> None:
        by = {a.article_key: a for a in corpus}
        # 第47条是经济补偿的核心条款，跨行合并错一次这里就红
        assert "经济补偿按劳动者在本单位工作的年限" in by["lcl#47"].text
        assert "每满一年支付一个月工资" in by["lcl#47"].text
        assert by["mfd#1000"].article_no == "第一千条"
        assert by["js1#54"].text.startswith("本解释自2021年1月1日起施行")

    def test_corpus_hash_is_deterministic_and_sensitive(self, corpus: list[Article]) -> None:
        assert corpus_hash(corpus) == corpus_hash(corpus)
        mutated = [a.model_copy(deep=True) for a in corpus]
        mutated[0].text += "改动"
        assert corpus_hash(mutated) != corpus_hash(corpus)
