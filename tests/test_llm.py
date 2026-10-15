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


# --------------------------------------------------------------------------- #
# 流式（F29 的前置：SSE 要边收边推，但缓存/重试这些"必须做的事"不能少）
# --------------------------------------------------------------------------- #

USAGE_CHUNK = {
    "choices": [],
    "usage": {
        "prompt_tokens": 300,
        "completion_tokens": 40,
        "total_tokens": 340,
        "prompt_cache_hit_tokens": 100,
        "prompt_cache_miss_tokens": 200,
        "completion_tokens_details": {"reasoning_tokens": 25},
    },
}


def evt(*chunks: dict | str) -> str:
    """拼若干数据行，**不补 `[DONE]`**。

    ⚠️ 每个 `data:` 后面必须跟一个空行——那是 SSE 的事件分隔符。
    少了空行，服务端会把相邻两个事件当成一个，解析出来是"少了一段答案"。
    这里用真实格式而不是"看起来像"的格式，就是为了让"漏了空行"这种错
    在测试里就暴露出来。
    """
    lines = []
    for c in chunks:
        payload = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        lines.append(f"data: {payload}\n\n")
    return "".join(lines)


def sse(*chunks: dict | str) -> str:
    """一段**完整**的事件流（末尾补 `[DONE]`）。

    `[DONE]` 在协议里是**终止符**：收到它就停止读流。
    所以想拼"两段流接起来"的测试必须用 `evt()`——用 `sse()`
    会在第一个 `[DONE]` 处断掉，而症状是"后半段内容凭空消失"，
    看起来像是解析器漏了，其实是测试自己把流截短了。
    """
    return evt(*chunks) + "data: [DONE]\n\n"


def delta(**kw) -> dict:
    """一个只有 delta 的 chunk。`choices` 的形状和真实响应一致。"""
    return {"choices": [{"delta": kw}]}


class _ScriptedStream(httpx.SyncByteStream):
    """故意在流中间断掉的响应体。**这是"吐出去一半就断了"唯一能造出来的方式**。"""

    def __init__(self, pieces: list[str], fail_at: int) -> None:
        self.pieces = pieces
        self.fail_at = fail_at

    def __iter__(self):
        for i, p in enumerate(self.pieces):
            if i == self.fail_at:
                raise httpx.ReadError("流断了")
            yield p.encode("utf-8")


class TestStreaming:
    def _client(
        self, tmp_path: Path, handler, *, mode: str | None = None
    ) -> tuple[LLMClient, list[dict]]:
        seen: list[dict] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return handler(request, len(seen) - 1)

        kw = {"mode": mode} if mode else {}
        c = LLMClient(
            api_key="sk-test-not-a-real-key",
            base_url="https://example.invalid",
            cache_dir=tmp_path / "cache",
            client=httpx.Client(transport=httpx.MockTransport(wrapped)),
            max_retries=2,
            **kw,
        )
        return c, seen

    def test_deltas_arrive_in_order_and_concatenate_to_the_text(
        self, tmp_path: Path
    ) -> None:
        body = sse(delta(content="试用期"), delta(content="最长"), delta(content="六个月。"))
        client, _ = self._client(
            tmp_path, lambda r, i: httpx.Response(200, content=body.encode("utf-8"))
        )

        got: list[str] = []
        resp = client.chat([{"role": "user", "content": "q"}], on_delta=got.append)

        assert got == ["试用期", "最长", "六个月。"]
        assert resp.text == "试用期最长六个月。"
        assert resp.text == "".join(got), "拼出来的正文必须等于 delta 的拼接，否则前端会少字"

    def test_request_asks_for_usage_and_stream(self, tmp_path: Path) -> None:
        """**必须显式要 usage。** 不带上它，流式响应的 usage 全是 0，
        成本报表会把每次问答都记成 0 元——而 0 和"没算"在报表上长得一样。"""
        body = sse(delta(content="答案"), USAGE_CHUNK)
        client, seen = self._client(
            tmp_path, lambda r, i: httpx.Response(200, content=body.encode("utf-8"))
        )
        client.chat([{"role": "user", "content": "q"}], on_delta=lambda _: None)

        assert seen[0]["stream"] is True
        assert seen[0]["stream_options"] == {"include_usage": True}

    def test_usage_is_read_from_the_trailing_chunk(self, tmp_path: Path) -> None:
        body = sse(delta(content="答案"), USAGE_CHUNK)
        client, _ = self._client(
            tmp_path, lambda r, i: httpx.Response(200, content=body.encode("utf-8"))
        )
        resp = client.chat([{"role": "user", "content": "q"}], on_delta=lambda _: None)

        assert resp.usage.prompt_tokens == 300
        assert resp.usage.completion_tokens == 40
        assert resp.usage.reasoning_tokens == 25
        assert resp.usage.prompt_cache_hit_tokens == 100

    def test_reasoning_deltas_go_to_their_own_callback(self, tmp_path: Path) -> None:
        """思考过程**不能混进正文**。混进去的话答案是
        "嗯，用户问的是试用期……试用期最长六个月"，用户看到一脸问号。"""
        body = sse(delta(reasoning_content="用户在问"), delta(content="六个月。"))
        client, _ = self._client(
            tmp_path, lambda r, i: httpx.Response(200, content=body.encode("utf-8"))
        )

        answer: list[str] = []
        think: list[str] = []
        resp = client.chat(
            [{"role": "user", "content": "q"}], on_delta=answer.append, on_reasoning=think.append
        )
        assert answer == ["六个月。"]
        assert think == ["用户在问"]
        assert resp.reasoning_text == "用户在问"

    def test_tool_call_fragments_are_stitched_together(self, tmp_path: Path) -> None:
        """流式下 tool_calls 是**碎片**：先给 id 和函数名，再一段段给参数。

        拼错的结果是参数 JSON 不完整 → 下游 `json.loads` 失败 →
        表现为"模型老是填不对参数"。**真相是我们没拼好**，不是模型不听话。
        """
        body = sse(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "calc_severance", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"months'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '_of_service": 36}'}}]}}]},
        )
        client, _ = self._client(
            tmp_path, lambda r, i: httpx.Response(200, content=body.encode("utf-8"))
        )
        resp = client.chat([{"role": "user", "content": "q"}], on_delta=lambda _: None)

        assert len(resp.tool_calls) == 1
        call = resp.tool_calls[0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "calc_severance"
        assert json.loads(call["function"]["arguments"]) == {"months_of_service": 36}

    def test_a_broken_chunk_does_not_throw_away_the_answer(self, tmp_path: Path) -> None:
        """一块坏数据不该让**已经收到的**答案作废。

        真实世界里这种块来自心跳、代理插入的注释、或半个被截断的 JSON。
        抛出去的结果是"用户等了 5 秒，然后收到一个错误"，
        而正确的做法是丢掉那一块、把剩下的答案给出去。
        """
        broken = (
            'data: {"choices": [{"delta": {"content"\n\n'
            "data: " + json.dumps(delta(content="中间"), ensure_ascii=False) + "\n\n"
        )
        body = (
            evt(delta(content="前半"))
            + broken
            + evt(delta(content="后半"))
            + "data: [DONE]\n\n"
        )
        client, _ = self._client(
            tmp_path, lambda r, i: httpx.Response(200, content=body.encode("utf-8"))
        )
        got: list[str] = []
        resp = client.chat([{"role": "user", "content": "q"}], on_delta=got.append)

        assert "前半" in resp.text and "后半" in resp.text
        assert got == ["前半", "中间", "后半"]

    def test_retries_when_absolutely_nothing_was_emitted(self, tmp_path: Path) -> None:
        """一个字都没吐出去时，重试是安全的。**这里 500 之后必须重试成功。**"""
        good = sse(delta(content="好的")).encode("utf-8")

        def handler(request: httpx.Request, i: int) -> httpx.Response:
            if i == 0:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, content=good)

        client, seen = self._client(tmp_path, handler)
        got: list[str] = []
        resp = client.chat([{"role": "user", "content": "q"}], on_delta=got.append)

        assert resp.text == "好的"
        assert got == ["好的"], "第一次失败什么都没吐，所以不该出现重复内容"
        assert len(seen) == 2

    def test_never_retries_once_a_single_delta_went_out(self, tmp_path: Path) -> None:
        """**这条是流式设计里最硬的一条约束。**

        已经吐出去的字用户已经看见了。这时候重试会把同一段话从头再吐一遍，
        用户看到的是"试用期最长……试用期最长不得超过六个月"。

        所以判据是 **"有没有字真的到过用户那边"**，而不是"这段流完整不完整"。
        """
        good = sse(delta(content="不该被用到"))
        pieces = [
            "data: " + json.dumps(delta(content="已经"), ensure_ascii=False) + "\n\n",
            "data: " + json.dumps(delta(content="吐出去了"), ensure_ascii=False) + "\n\n",
        ]

        def handler(request: httpx.Request, i: int) -> httpx.Response:
            if i == 0:
                return httpx.Response(200, stream=_ScriptedStream(pieces, fail_at=1))
            return httpx.Response(200, content=good.encode("utf-8"))

        client, seen = self._client(tmp_path, handler)
        got: list[str] = []
        with pytest.raises(LLMError):
            client.chat([{"role": "user", "content": "q"}], on_delta=got.append)

        assert got == ["已经"], "只应看到断流之前吐出去的那部分"
        assert len(seen) == 1, "已经吐过字了就不许再发一次请求"

    def test_truncated_stream_escalates_budget_and_stays_streaming(
        self, tmp_path: Path
    ) -> None:
        """推理模型预算不够时会返回"空正文但烧了 token"。

        非流式路径会自动翻倍预算重试，流式路径**也必须**——
        而且升级之后**仍然是流式**。早期版本这里写死了走非流式，
        表现是"预算不够时用户那边的流突然断了、答案一次蹦出来"，
        看起来像流式坏了，不像预算升级了。
        """
        starved = sse({
            "choices": [{"delta": {}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 32,
                "total_tokens": 132,
                "completion_tokens_details": {"reasoning_tokens": 32},
            },
        })
        good = sse(delta(content="六个月。"))

        def handler(request: httpx.Request, i: int) -> httpx.Response:
            body = starved if i == 0 else good
            return httpx.Response(200, content=body.encode("utf-8"))

        client, seen = self._client(tmp_path, handler)
        got: list[str] = []
        resp = client.chat(
            [{"role": "user", "content": "q"}], max_tokens=32, on_delta=got.append
        )

        assert len(seen) == 2
        assert seen[0]["max_tokens"] == 32
        assert seen[1]["max_tokens"] == 128, "预算是翻了 4 倍"
        assert seen[1]["stream"] is True, "升级之后必须还是流式"
        assert resp.text == "六个月。"
        assert got == ["六个月。"]

    def test_cache_hit_still_produces_one_delta(self, tmp_path: Path) -> None:
        """缓存命中时"逐段吐"没有意义（本来就是一瞬读完的），
        但**必须给一个 delta 事件**——否则前端会转半天圈、然后整段答案突然出现，
        看起来像页面卡了，而不像"这一轮是从缓存来的"。"""
        body = sse(delta(content="缓存里的答案"))
        client, _ = self._client(
            tmp_path,
            lambda r, i: httpx.Response(200, content=body.encode("utf-8")),
            mode="record",
        )
        client.chat([{"role": "user", "content": "q"}], on_delta=lambda _: None)

        replay = LLMClient(
            api_key="sk-test-not-a-real-key",
            base_url="https://example.invalid",
            cache_dir=tmp_path / "cache",
            client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b""))
            ),
            mode="replay",
        )
        got: list[str] = []
        resp = replay.chat([{"role": "user", "content": "q"}], on_delta=got.append)
        assert resp.from_cache is True
        assert got == ["缓存里的答案"]
