"""LLM 客户端的测试。**不打网络**——用 `httpx.MockTransport` 造响应。

这里钉的是四件在真实调用里吃过亏的事：

1. **推理模型的空正文**：`max_tokens` 覆盖思维链，给小了会返回空串且不报错。
   客户端必须自动加预算重试，而不是把空串交给下游去 `json.loads`。
2. **4xx 不重试**：参数错/鉴权错重试只会重犯，顺便烧额度。
3. **429/5xx 才重试**。
4. **缓存键必须对影响输出的入参敏感**，且 record/replay 行为要可断言——
   否则"离线复现"这个承诺是空的。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from juris_agent.llm import (
    LLMAuthError,
    LLMBadRequest,
    LLMClient,
    LLMError,
    LLMServerError,
    LLMTruncated,
    ReplayMiss,
    Usage,
    _cache_key,
)


def ok_body(text: str = "回答", *, completion: int = 10, reasoning: int = 0) -> dict:
    return {
        "model": "deepseek-flash",
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion,
            "total_tokens": 100 + completion,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class _Recorder:
    """记录每次请求，并按脚本给出响应。"""

    def __init__(self, script: list[tuple[int, dict | str]]) -> None:
        self.script = script
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        status, body = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(body, str):
            return httpx.Response(status, content=body.encode("utf-8"))
        return httpx.Response(status, json=body)


def make_client(tmp_path: Path, script, **kw) -> tuple[LLMClient, _Recorder]:
    rec = _Recorder(script)
    client = LLMClient(
        api_key="sk-test-not-a-real-key",
        base_url="https://example.invalid",
        cache_dir=tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(rec)),
        max_retries=kw.pop("max_retries", 2),
        **kw,
    )
    return client, rec


class TestUsage:
    def test_addition_is_fieldwise(self) -> None:
        a = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3, reasoning_tokens=1)
        b = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, reasoning_tokens=5)
        c = a + b
        assert c.prompt_tokens == 11 and c.completion_tokens == 22
        assert c.total_tokens == 33 and c.reasoning_tokens == 6


class TestCacheKey:
    def test_deterministic(self) -> None:
        p = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.0, "max_tokens": 10}
        assert _cache_key(p) == _cache_key(dict(p))

    @pytest.mark.parametrize(
        "field,value",
        [("model", "other"), ("temperature", 1.0), ("max_tokens", 99)],
    )
    def test_sensitive_to_output_affecting_params(self, field: str, value: object) -> None:
        base = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.0, "max_tokens": 10}
        changed = {**base, field: value}
        assert _cache_key(base) != _cache_key(changed)

    def test_message_order_matters(self) -> None:
        a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 10}
        b = {"model": "m", "messages": [{"role": "user", "content": "y"}], "max_tokens": 10}
        assert _cache_key(a) != _cache_key(b)


class TestTruncation:
    """推理模型的坑：思维链吃光 max_tokens → 空正文。"""

    def test_empty_body_with_tokens_is_detected(self, tmp_path: Path) -> None:
        client, _ = make_client(tmp_path, [(200, ok_body("", completion=32, reasoning=32))], max_retries=0)
        with pytest.raises(LLMTruncated):
            client.chat([{"role": "user", "content": "hi"}], max_tokens=32)

    def test_auto_escalates_budget_and_retries(self, tmp_path: Path) -> None:
        """第一次空正文 → 自动把预算翻 4 倍再来一次。**不该把空串抛给调用方。**"""
        client, rec = make_client(
            tmp_path,
            [(200, ok_body("", completion=32, reasoning=32)), (200, ok_body("正常回答"))],
            max_retries=0,
        )
        resp = client.chat([{"role": "user", "content": "hi"}], max_tokens=32)
        assert resp.text == "正常回答"
        assert len(rec.requests) == 2
        assert rec.requests[0]["max_tokens"] == 32
        assert rec.requests[1]["max_tokens"] == 128

    def test_escalation_is_capped(self, tmp_path: Path) -> None:
        client, rec = make_client(
            tmp_path,
            [(200, ok_body("", completion=1)), (200, ok_body("好了"))],
            max_retries=0,
        )
        client.chat([{"role": "user", "content": "hi"}], max_tokens=20000)
        assert rec.requests[1]["max_tokens"] == 32768  # 不无限翻倍

    def test_whitespace_only_body_counts_as_empty(self, tmp_path: Path) -> None:
        client, _ = make_client(tmp_path, [(200, ok_body("   \n  ", completion=5))], max_retries=0)
        with pytest.raises(LLMTruncated):
            client.chat([{"role": "user", "content": "hi"}], max_tokens=32)

    def test_genuinely_empty_with_zero_tokens_is_fine(self, tmp_path: Path) -> None:
        """completion_tokens=0 的空回复是另一回事（模型没说话），不当作截断。"""
        client, _ = make_client(tmp_path, [(200, ok_body("", completion=0))], max_retries=0)
        assert client.chat([{"role": "user", "content": "hi"}]).text == ""


class TestErrorClassification:
    def test_401_is_auth_error_not_retried(self, tmp_path: Path) -> None:
        client, rec = make_client(tmp_path, [(401, {"error": "bad key"})])
        with pytest.raises(LLMAuthError):
            client.chat([{"role": "user", "content": "hi"}])
        assert len(rec.requests) == 1, "鉴权失败不该重试"

    def test_400_is_bad_request_not_retried(self, tmp_path: Path) -> None:
        client, rec = make_client(tmp_path, [(400, {"error": "invalid model"})])
        with pytest.raises(LLMBadRequest):
            client.chat([{"role": "user", "content": "hi"}])
        assert len(rec.requests) == 1

    def test_500_is_retried_then_raises(self, tmp_path: Path) -> None:
        client, rec = make_client(tmp_path, [(500, "boom")], max_retries=2)
        with pytest.raises(LLMServerError):
            client.chat([{"role": "user", "content": "hi"}])
        assert len(rec.requests) == 3  # 首次 + 2 次重试

    def test_429_then_success(self, tmp_path: Path) -> None:
        client, rec = make_client(
            tmp_path, [(429, "slow down"), (200, ok_body("终于好了"))], max_retries=2
        )
        assert client.chat([{"role": "user", "content": "hi"}]).text == "终于好了"
        assert len(rec.requests) == 2

    def test_no_key_is_rejected_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """没配 key 时要在**构造期**就拦住，而不是等到第一次调用才炸。

        这里必须把 settings 也一起打掉：`LLMClient` 在 `api_key` 为空时会
        回落到 `Settings`，而开发机上 `.env` 通常是有 key 的，
        不打掉就永远测不到"真的没有 key"这条路径。
        """
        from juris_agent import llm as llm_mod
        from juris_agent.config import Settings

        # 必须打在 `juris_agent.llm` 上——它是 `from ... import get_settings`，
        # 拿到的是函数对象的引用，打 `config.get_settings` 对它没有影响。
        monkeypatch.setattr(
            llm_mod, "get_settings",
            lambda: Settings(_env_file=None, deepseek_api_key=None),
        )
        with pytest.raises(LLMAuthError):
            LLMClient(api_key="")


class TestUsageAccounting:
    def test_accumulates_across_calls(self, tmp_path: Path) -> None:
        client, _ = make_client(tmp_path, [(200, ok_body(completion=10))])
        client.chat([{"role": "user", "content": "a"}])
        client.chat([{"role": "user", "content": "b"}])
        assert client.usage_total.prompt_tokens == 200
        assert client.usage_total.completion_tokens == 20

    def test_cached_response_adds_no_tokens(self, tmp_path: Path) -> None:
        """缓存命中的那条**不该再计一次费用**，否则成本报表会高估。"""
        client, rec = make_client(tmp_path, [(200, ok_body())], mode="record")
        client.chat([{"role": "user", "content": "same"}])
        first = client.usage_total.prompt_tokens
        client.chat([{"role": "user", "content": "same"}])
        assert client.usage_total.prompt_tokens == first
        assert len(rec.requests) == 1


class TestRecordReplay:
    def test_record_then_replay_makes_no_new_request(self, tmp_path: Path) -> None:
        c1, rec1 = make_client(tmp_path, [(200, ok_body("第一次"))], mode="record")
        assert c1.chat([{"role": "user", "content": "q"}]).text == "第一次"
        c2, rec2 = make_client(tmp_path, [(200, ok_body("不该被用到"))], mode="replay")
        assert c2.chat([{"role": "user", "content": "q"}]).text == "第一次"
        assert rec2.requests == []

    def test_replay_miss_raises_loudly(self, tmp_path: Path) -> None:
        """**静默回落到真实 API 会让"离线复现"变成谎话**，所以必须抛错。"""
        client, rec = make_client(tmp_path, [(200, ok_body())], mode="replay")
        with pytest.raises(ReplayMiss):
            client.chat([{"role": "user", "content": "从没记录过"}])
        assert rec.requests == []

    def test_off_mode_writes_nothing(self, tmp_path: Path) -> None:
        client, _ = make_client(tmp_path, [(200, ok_body())], mode="off")
        client.chat([{"role": "user", "content": "q"}])
        assert list((tmp_path / "cache").glob("*.json")) == []

    def test_replayed_response_is_marked(self, tmp_path: Path) -> None:
        c1, _ = make_client(tmp_path, [(200, ok_body())], mode="record")
        c1.chat([{"role": "user", "content": "q"}])
        c2, _ = make_client(tmp_path, [(200, ok_body())], mode="replay")
        assert c2.chat([{"role": "user", "content": "q"}]).from_cache is True


class TestCompleteJson:
    def test_parses_json(self, tmp_path: Path) -> None:
        client, rec = make_client(tmp_path, [(200, ok_body('{"a": 1}'))])
        assert client.complete_json("给我个 json") == {"a": 1}
        # 必须带上 response_format，让服务端强制合法 JSON
        assert rec.requests[0]["response_format"] == {"type": "json_object"}

    def test_bad_json_raises_not_returns_none(self, tmp_path: Path) -> None:
        """**静默返回 None 会让一批题目悄悄变成垃圾数据**，必须抛。"""
        client, _ = make_client(tmp_path, [(200, ok_body("这不是 json"))])
        with pytest.raises(LLMError, match="不是合法 JSON"):
            client.complete_json("给我个 json")


class TestCompleteMany:
    def test_preserves_input_order(self, tmp_path: Path) -> None:
        """顺序必须保——gold 写回时错位是**静默**的。"""
        rec = _Recorder([(200, ok_body("x"))])

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            text = body["messages"][-1]["content"]
            return httpx.Response(200, json=ok_body(f"回答-{text}"))

        client = LLMClient(
            api_key="sk-test", base_url="https://example.invalid",
            cache_dir=tmp_path / "c",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        out = client.complete_many(
            [{"messages": [{"role": "user", "content": str(i)}]} for i in range(8)],
            concurrency=4,
        )
        assert [r.text for r in out] == [f"回答-{i}" for i in range(8)]

    def test_empty_input(self, tmp_path: Path) -> None:
        client, _ = make_client(tmp_path, [(200, ok_body())])
        assert client.complete_many([]) == []
