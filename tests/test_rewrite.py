"""F9 查询改写的测试。

最重要的一条纪律：**原始 query 永远排在变体列表第一位**。
改写是扩召回不是替换，所以"改写跑偏"最坏只会退化成"没改写"，
不会让结果比不改更差。这条一旦破了，`+改写` 那行就可能反而变差，
而你会去别处找原因。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from juris_agent.retrieval.rewrite import (
    IdentityRewriter,
    LLMRewriter,
    SynonymRewriter,
    SynonymTable,
)


@pytest.fixture
def table() -> SynonymTable:
    return SynonymTable(
        version="test",
        map={
            "辞职": ["解除劳动合同"],
            "老板": ["用人单位"],
            "加班": ["延长工作时间", "延长工时"],
        },
    )


class TestSynonymTable:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "s.yaml"
        p.write_text("version: '2'\nmap:\n  辞职:\n    - 解除劳动合同\n", encoding="utf-8")
        t = SynonymTable.load(p)
        assert t.version == "2"
        assert t.map["辞职"] == ["解除劳动合同"]

    def test_missing_file_is_empty_not_fatal(self, tmp_path: Path) -> None:
        t = SynonymTable.load(tmp_path / "不存在.yaml")
        assert t.map == {}


class TestIdentityRewriter:
    def test_returns_query_unchanged(self) -> None:
        assert IdentityRewriter().rewrite("随便问问") == ["随便问问"]


class TestSynonymRewriter:
    def test_no_match_returns_query_only(self, table: SynonymTable) -> None:
        assert SynonymRewriter(table).rewrite("竞业限制几年") == ["竞业限制几年"]

    def test_original_is_always_first(self, table: SynonymTable) -> None:
        out = SynonymRewriter(table).rewrite("我要辞职")
        assert out[0] == "我要辞职"

    def test_generates_replacement_variant(self, table: SynonymTable) -> None:
        out = SynonymRewriter(table).rewrite("我要辞职")
        assert "我要解除劳动合同" in out

    def test_multiple_terms_produce_multiple_variants(self, table: SynonymTable) -> None:
        out = SynonymRewriter(table).rewrite("老板不让我辞职")
        assert "用人单位不让我辞职" in out
        assert "老板不让我解除劳动合同" in out

    def test_terms_that_expand_to_several(self, table: SynonymTable) -> None:
        out = SynonymRewriter(table).rewrite("天天加班")
        assert "天天延长工作时间" in out
        assert "天天延长工时" in out

    def test_respects_max_variants(self, table: SynonymTable) -> None:
        out = SynonymRewriter(table, max_variants=3).rewrite("老板让我加班后辞职")
        assert len(out) == 3

    def test_variants_are_deduplicated(self) -> None:
        # 两个不同的口语词映射到同一个术语 → 替换后得到同一个字符串
        t = SynonymTable(map={"A": ["X"], "B": ["X"]})
        out = SynonymRewriter(t).rewrite("AB")
        assert len(out) == len(set(out))

    def test_deterministic_order(self, table: SynonymTable) -> None:
        """同样输入两次必须产生同样顺序的变体——否则检索结果不可复现。"""
        r = SynonymRewriter(table)
        q = "老板让我加班后辞职"
        assert r.rewrite(q) == r.rewrite(q)

    def test_match_reports_hits_in_text_order(self, table: SynonymTable) -> None:
        hits = SynonymRewriter(table).match("老板让我辞职")
        assert [w for w, _ in hits] == ["老板", "辞职"]


class _FakeLLM:
    """假客户端：不发网络请求，直接吐出预设 JSON。"""

    def __init__(self, payload: object, raise_exc: bool = False) -> None:
        self.payload = payload
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    def complete_json(self, prompt: str, system: str | None = None, **kw: object) -> object:
        self.calls.append(prompt)
        if self.raise_exc:
            raise RuntimeError("模拟模型返回垃圾")
        return self.payload


class TestLLMRewriter:
    def test_original_first_then_variants(self) -> None:
        llm = _FakeLLM({"variants": ["解除劳动合同 提前三十日", "劳动者 单方解除"]})
        out = LLMRewriter(llm).rewrite("我想辞职怎么办")
        assert out[0] == "我想辞职怎么办"
        assert "解除劳动合同 提前三十日" in out

    def test_dedupes_when_model_echoes_query(self) -> None:
        llm = _FakeLLM({"variants": ["我想辞职怎么办", "解除劳动合同"]})
        out = LLMRewriter(llm).rewrite("我想辞职怎么办")
        assert out.count("我想辞职怎么办") == 1

    def test_bad_payload_degrades_not_raises(self) -> None:
        """模型回非 JSON 时退化成不改写，**但不抛错**——改写是增强不是必需。"""
        llm = _FakeLLM(None, raise_exc=True)
        r = LLMRewriter(llm)
        assert r.rewrite("问一句") == ["问一句"]

    def test_failure_is_counted(self) -> None:
        """降级必须记账。否则"改写收益"里混着一批根本没改写的题而你不知道。"""
        r = LLMRewriter(_FakeLLM(None, raise_exc=True))
        r.rewrite("a")
        r.rewrite("b")
        assert r.failures == 2

    def test_non_list_variants_degrades(self) -> None:
        r = LLMRewriter(_FakeLLM({"variants": "不是列表"}))
        assert r.rewrite("问") == ["问"]
        assert r.failures == 1

    def test_respects_max_variants(self) -> None:
        llm = _FakeLLM({"variants": [f"v{i}" for i in range(20)]})
        assert len(LLMRewriter(llm, max_variants=4).rewrite("q")) == 4

    def test_blank_variants_dropped(self) -> None:
        llm = _FakeLLM({"variants": ["  ", "", "有用"]})
        out = LLMRewriter(llm).rewrite("q")
        assert out == ["q", "有用"]
