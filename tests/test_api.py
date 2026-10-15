"""HTTP 层（F29）的测试。

## 这一层测的是什么

**测**：接口契约——状态码对不对、事件流的事件名和顺序对不对、
会话有没有串、版本号冲突拦不拦得住、**状态从"图"搬到"响应"的过程中
字段有没有静默丢掉**。

**不测**：答案质量（那是评测层的事，见 `tests/test_agent_graph.py` 的说明）。
这里全程用 `FakeRetriever` + `ScriptedLLM`，**不加载模型、不发网络请求**。

## 为什么这一层的测试特别值

因为它测的是**最容易静默出错的一段代码**：

图的输出 → 响应模型 → JSON → 前端。这条路上每一步都在"转格式"，
而转格式出错的方式是**丢字段**，不是抛异常。接口照样返回 200，
前端拿到一个空字符串，用户看到一条空白回答。
**没有任何异常、没有日志、没有红字。**

所以这一层的断言偏向"某个字段确实带着值到了对面"，
而不是"函数被调用了"。后面 `test_values_state_is_a_dict_...`
就是这类——它是我在真机跑通之后回头补的，因为**实测发现
`values` 模式给的是 dict 而不是 `AgentState`**，
而按对象来读会一路静默返回默认值。
"""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from juris_agent.agent.graph import make_runner
from juris_agent.api.app import create_app
from juris_agent.api.runtime import Runtime
from juris_agent.api.schemas import ChatResponse
from juris_agent.api.sessions import SessionConflict, SessionStore
from juris_agent.config import Settings
from juris_agent.tools.builtin import build_registry
from juris_agent.tools.executor import ToolExecutor
from tests.fakes import FakeRetriever, ScriptedLLM, tool_call

ANSWER = (
    "《劳动合同法》第四十七条 经济补偿按劳动者在本单位工作的年限，"
    "每满一年支付一个月工资的标准向劳动者支付。"
)


# --------------------------------------------------------------------------- #
# 搭台子
# --------------------------------------------------------------------------- #


def _settings() -> Settings:
    return Settings(env="prod", deepseek_api_key=None)


def _runner(script: list, retriever=None):
    """一个**完全不碰外部世界**的图。

    注意脚本长度：一次问答最多会调**三次**模型
    （工具循环里两次 + `generate` 一次），所以写两段脚本时最后一轮
    大概率拿到空正文——表现为 `answer=""`、`finish_reason="refused"`。
    **这不是 bug，是脚本没写够。** 调试时先数模型调用次数，别急着改代码。
    """
    s = _settings()
    r = retriever if retriever is not None else FakeRetriever()
    llm = ScriptedLLM(script)
    reg = build_registry(r)
    ex = ToolExecutor(reg, s)
    run = make_runner(retriever=r, registry=reg, executor=ex, llm=llm, settings=s)
    return run, r, llm, s


def _app(script: list, *, serve_web: bool = False, retriever=None, runner=None, llm=None, s=None):
    """造一个注入了假依赖的 app。

    ⚠️ **注入是唯一的做法，不是"可选的灵活性"。** `create_app()` 不带参数时
    会在 lifespan 里装配真检索器（加载 1GB 模型、还要 API key）。
    只要有一个测试忘了注入，整套测试就会从 11 秒变成几分钟——
    然后所有人都开始用 `-k` 挑测试跑，CI 也就名存实亡了。
    """
    if runner is None:
        # ⚠️ **必须把 `retriever` 传下去。** 一开始漏了这一步，
        # 于是传进来的检索器被丢掉、永远用默认的 `FakeRetriever`，
        # 而默认那个**没有 `articles_by_key`**——`/api/article` 全返回 503。
        # 这种"参数收下了但没用"的 bug 不会报错，只会让一整组测试
        # 悄悄测不到它们想测的东西。
        runner, retriever, llm, s = _runner(script, retriever=retriever)
    rt = Runtime(
        settings=s or _settings(),
        sessions=SessionStore(new_memory=lambda: None),
        runner=runner,
        retriever=retriever,
        llm=llm,
    )
    return create_app(rt, serve_web=serve_web), rt


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _frames(text: str) -> list[tuple[str, dict]]:
    """把 SSE 原文切成 `(事件名, 数据)` 列表。

    ⚠️ 断言的是**原始文本**切出来的东西，不是"我以为服务端发了什么"。
    如果只断言 handler 里构造的 `Event` 对象，
    那么"`to_sse()` 忘了两个换行"、"中文被转义成 `\\uXXXX`"这类
    **只发生在渲染环节**的问题就永远测不到——而那正好是前端会踩的坑。
    """
    out: list[tuple[str, dict]] = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        out.append((name, json.loads(data) if data else {}))
    return out


# --------------------------------------------------------------------------- #
# ① 健康检查
# --------------------------------------------------------------------------- #


class TestHealth:
    async def test_ready_when_runner_present(self):
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            h = (await c.get("/api/health")).json()
        assert h["ok"] is True
        assert h["ready"] is True
        assert h["problems"] == []

    async def test_not_ready_when_assembly_failed(self):
        """**"服务在跑"和"答得了问题"是两件事**，接口必须分开报。

        否则前端只能显示"网络错误"，而真正的原因（少个 API key）
        只留在服务端日志里。
        """
        app, _ = _app([], runner=None, retriever=FakeRetriever(), llm=None)
        app.state.runtime.runner = None
        app.state.runtime.problems = ["LLM 未装配：没有 API key"]
        async with _client(app) as c:
            h = (await c.get("/api/health")).json()
        assert h["ready"] is False
        assert h["problems"] == ["LLM 未装配：没有 API key"]

    async def test_chat_is_503_not_500_when_not_ready(self):
        """没就绪是 **503**（暂时不可用）不是 500（服务端出错）。

        区别是有意义的：503 是在说"等运维把配置补上就能用"，
        500 是在说"这次请求把代码跑挂了"。压测脚本和监控
        通常按状态码分类，混用会让它们报出错误的结论。
        """
        app, _ = _app([], runner=None, retriever=FakeRetriever(), llm=None)
        app.state.runtime.runner = None
        app.state.runtime.problems = ["检索器未装配"]
        async with _client(app) as c:
            r = await c.post("/api/chat", json={"question": "试用期多久"})
        assert r.status_code == 503
        assert "检索器未装配" in json.dumps(r.json(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# ② 非流式问答
# --------------------------------------------------------------------------- #


class TestChat:
    async def test_happy_path_carries_answer_through(self):
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            r = await c.post("/api/chat", json={"question": "经济补偿怎么算"})
        assert r.status_code == 200
        d = r.json()
        # **这条断言的重点不是"等于 ANSWER"，而是"答案没在转格式的路上被丢掉"。**
        # 用 `in` 而不是 `==`，是因为引用校验可能给编号加标记。
        assert ANSWER[:20] in d["answer"]
        assert d["finish_reason"] == "answered"
        assert d["session_id"].startswith("s-")
        assert d["turn"] == 1

    async def test_response_matches_declared_schema(self):
        """响应必须能过 `ChatResponse` 的校验。

        为什么单拎一条测：`/api/chat` 挂了 `response_model=ChatResponse`，
        **但 `/api/chat/stream` 的 `done` 事件没有**——它是手写 dict 发出去的。
        也就是说"两条路的响应字段可能不一致"，而只有这条测试能保证
        非流式那条路是真的合规。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            r = await c.post("/api/chat", json={"question": "经济补偿怎么算"})
        ChatResponse.model_validate(r.json())  # 不抛就算过

    async def test_second_turn_reuses_session(self):
        app, _ = _app([ANSWER] * 6)
        async with _client(app) as c:
            a = (await c.post("/api/chat", json={"question": "第一问"})).json()
            b = (
                await c.post(
                    "/api/chat",
                    json={"question": "第二问", "session_id": a["session_id"]},
                )
            ).json()
        assert b["session_id"] == a["session_id"]
        assert b["turn"] == 2, "轮次没有累加，说明会话没接上"

    async def test_trace_and_cost_present(self):
        """成本与 trace 必须真的到了响应里。

        它们是最容易"算出来了但忘了往响应里塞"的两个字段：
        计算逻辑在别的模块（`observability/cost.py`、节点的 trace），
        响应模型里漏掉一个赋值**不会报错**，只是永远空着。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()
        assert d["cost"] is not None
        assert d["cost"]["as_of"] != "unknown", "定价表没读到，成本报表会缺一个基准日"
        assert len(d["trace"]) >= 3, f"trace 步数太少：{len(d['trace'])}"
        assert {"remember", "entry", "retrieve"} <= {t["node"] for t in d["trace"]}

    async def test_client_history_is_accepted(self):
        """请求里带 `history` 时用调用方的，而不是服务端的。

        用途是"无状态调用方"：curl / 脚本 / 别的服务想自己管历史，
        不想依赖服务端的会话表。**这条路必须真的通**，
        否则文档里写的那句"可以不传 session_id"就是假的。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            r = await c.post(
                "/api/chat",
                json={
                    "question": "那我的情况呢",
                    "history": [
                        {"role": "user", "content": "我月薪 12000"},
                        {"role": "assistant", "content": "好的"},
                    ],
                },
            )
        assert r.status_code == 200, r.text

    async def test_empty_question_is_rejected(self):
        """空问题在**入口**就被挡掉，不要送到模型那里花一次钱。"""
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            r = await c.post("/api/chat", json={"question": ""})
        assert r.status_code == 422


# --------------------------------------------------------------------------- #
# ③ 流式问答（SSE）
# --------------------------------------------------------------------------- #


class TestStream:
    async def test_events_arrive_in_pipeline_order(self):
        """事件顺序 = 链路顺序。**这是"可溯源"能被看见的前提。**

        断言用"子序列"而不是"完全相等"：
        加了新的中间事件不该让这条测试红（否则每次加事件都要改测试，
        改着改着就没人认真看断言了）。但**关键四步的相对顺序必须钉住**，
        因为前端是按这个顺序画时间线的。
        """
        app, _ = _app(
            [
                {"tool_calls": [tool_call("c1", "search_law", {"query": "经济补偿"})]},
                ANSWER,
                ANSWER,
            ]
        )
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "经济补偿怎么算"}
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                text = "".join([chunk async for chunk in resp.aiter_text()])

        names = [n for n, _ in _frames(text)]
        for want in ("memory", "route", "retrieval", "tool_call", "tool_result", "done"):
            assert want in names, f"事件流里没有 {want}：{names}"
        # 相对顺序：记忆 → 路由 → 检索 → 工具 → 收尾
        assert names.index("memory") < names.index("route") < names.index("retrieval")
        assert names.index("tool_call") < names.index("tool_result") < names.index("done")
        assert names[-1] == "done", "最后一个事件必须是 done"

    async def test_done_carries_authoritative_answer(self):
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "经济补偿怎么算"}
            ) as resp:
                text = "".join([chunk async for chunk in resp.aiter_text()])
        done = dict(_frames(text))["done"]
        assert ANSWER[:20] in done["answer"]
        assert done["session_id"].startswith("s-")
        assert done["cost"] is not None

    async def test_tool_result_carries_the_calculation_steps(self):
        """`tool_result` 事件里必须带着**算式与代入过程**。

        计算题只给结论（"能拿 24000"）的话，用户没有任何办法核对它对不对。
        工具算出来的 `steps` 是"可核验"的全部来源——
        和后端那句"计算题必须给算式与代入过程"是一件事的两头。

        ⚠️ 这条测试**是被一次真实的疏漏逼出来的**：后端一直在算 `steps`、
        事件也一直在传 `payload`，但**界面没渲染它**，只显示了
        "✅ calc_severance ok 12ms"。所以这里断言的是**事件的契约**
        （payload 里有没有 steps），而页面有没有把它画出来由
        `TestWeb` 里那条字符串检查兜着——两件事分开测，
        因为它们会各自坏掉，坏掉的原因也不一样。
        """
        app, _ = _app(
            [
                {"tool_calls": [tool_call("c1", "calc_severance", {
                    # ⚠️ 枚举值必须写对。第一版我写了 `unlawful`（凭印象编的），
                    # 结果入参校验直接失败 → 工具返回 failed → `payload` 是 None，
                    # 测试红在"payload 是空的"。**这正是入参校验在干活的样子**：
                    # 编一个自以为合理的枚举值，会被当场拦住，而不是算出一个数。
                    "cause": "illegal_dismissal", "months_of_service": 38,
                    "avg_monthly_wage": 12000,
                })]},
                ANSWER,
                ANSWER,
            ]
        )
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "被无故辞退能拿多少"}
            ) as resp:
                text = "".join([chunk async for chunk in resp.aiter_text()])

        results = [data for name, data in _frames(text) if name == "tool_result"]
        assert results, "这一轮没有 tool_result 事件"
        payload = results[-1].get("payload")
        assert payload, "tool_result 的 payload 是空的，前端拿不到算式"
        assert results[-1]["status"] == "ok", f"工具没成功：{results[-1].get('error')}"
        steps = payload.get("steps") or []
        assert len(steps) >= 2, f"算式步骤太少（{steps!r}），不足以让人核对"
        # 步骤是**给人看的句子**，不是字段名。断言这一点是因为
        # "把原始 dict 塞进去"也能让 `len(steps) >= 2` 成立。
        assert any("月" in str(s) and ("×" in str(s) or "*" in str(s) or "=" in str(s))
                   for s in steps), f"步骤里看不出怎么算的：{steps!r}"

    async def test_sse_frames_are_not_ascii_escaped(self):
        """中文必须原样出去，**不能变成 `\\u52b3\\u52a8`**。

        理由很实际：`curl` 看事件流是这个接口的验收方式之一，
        而一屏 `\\uXXXX` 虽然能跑通、却没法演示。
        `json.dumps` 默认就转义，所以这条很容易被漏掉。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "经济补偿怎么算"}
            ) as resp:
                text = "".join([chunk async for chunk in resp.aiter_text()])
        assert "\\u" not in text, "SSE 里有 unicode 转义，curl 出来是一堆乱码"
        assert "劳动" in text

    async def test_frames_end_with_blank_line(self):
        """每帧必须两个换行结尾，否则前端会一直等下一帧。

        表现是"事件全收到了、界面就是不更新"——因为解析器在等一个
        永远不来的空行。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "经济补偿怎么算"}
            ) as resp:
                text = "".join([chunk async for chunk in resp.aiter_text()])
        assert text.endswith("\n\n")

    async def test_post_and_get_share_one_generator(self):
        """POST 和 GET 两条路必须给出**同样形状**的事件流。

        分别实现的话，迟早出现"GET 能用的字段 POST 没有"这种
        只在一条路上存在的差异，而演示时用的是 GET、前端用的是 POST。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "经济补偿怎么算"}
            ) as r1:
                get_names = [n for n, _ in _frames("".join([x async for x in r1.aiter_text()]))]
            async with c.stream(
                "POST", "/api/chat/stream", json={"question": "经济补偿怎么算"}
            ) as r2:
                post_names = [n for n, _ in _frames("".join([x async for x in r2.aiter_text()]))]
        assert get_names == post_names

    async def test_retrieval_event_carries_article_text_inline(self):
        """法条正文跟着检索事件一起到。

        前端"点开看原文"就靠它，不需要再发一次请求。
        这条断言防的是"以后有人为了省带宽把 text 去掉"——
        那会让点开动作多一次往返，明显发顿。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "经济补偿怎么算"}
            ) as resp:
                text = "".join([chunk async for chunk in resp.aiter_text()])
        ret = dict(_frames(text))["retrieval"]
        assert ret["count"] >= 1
        assert ret["articles"][0]["text"].strip(), "法条正文没带出来，点开就是空的"

    async def test_no_runner_streams_an_error_event_not_a_broken_socket(self):
        """装配失败时**用事件说**，而不是把连接直接掐掉。

        掐掉的话前端只能显示"网络错误"，看不到真正缺什么配置。
        """
        app, _ = _app([], runner=None, retriever=FakeRetriever(), llm=None)
        app.state.runtime.runner = None
        app.state.runtime.problems = ["没有 API key"]
        async with _client(app) as c:
            async with c.stream(
                "GET", "/api/chat/stream", params={"question": "试用期多久"}
            ) as resp:
                text = "".join([chunk async for chunk in resp.aiter_text()])
        frames = dict(_frames(text))
        assert "error" in frames
        assert "没有 API key" in json.dumps(frames["error"], ensure_ascii=False)


# --------------------------------------------------------------------------- #
# ④ 会话与并发：这一层最容易出"静默覆盖"
# --------------------------------------------------------------------------- #


class TestSessions:
    async def test_unknown_session_id_creates_fresh_one(self):
        """给了个不认识的 id 就建新的，**不报错**。

        理由：客户端可能存着一个服务端已淘汰（TTL/LRU）的 id。
        这种情况报错会逼着用户"刷新一下重来"，而正确行为是
        "当作新会话继续"——丢的只是上下文，不是服务可用性。
        """
        app, rt = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "问一句", "session_id": "s-nope"})).json()
        assert d["session_id"] == "s-nope"
        assert rt.sessions.stats()["sessions"] == 1

    def test_commit_rejects_stale_version(self):
        """CAS：读时 version=0，写之前已经是 1 → **必须抛，不能覆盖**。

        这是本模块最重要的一条。它防的是"多进程部署"和"某条新路径忘了加锁"
        这两种锁管不着的情况，而它们造成的后果是一样的：
        **用户说的话被静默吞掉**。
        """
        store = SessionStore(new_memory=lambda: None)
        s = store.get_or_create(None)
        stale = s.version  # 读到的版本号
        store.commit(s, expected_version=stale, question="第一句", answer="第一答")
        with pytest.raises(SessionConflict):
            store.commit(s, expected_version=stale, question="第二句", answer="第二答")
        assert [m["content"] for m in s.history] == ["第一句", "第一答"], "被覆盖了"

    def test_messages_returns_a_copy(self):
        """`messages()` 必须返回拷贝。

        返回原 list 的话，图里任何一次 `append` 都会改到会话本身，
        下一轮历史里就多一条重复消息——**症状是"模型好像看到两遍同一个问题"，
        从日志上根本看不出来**。
        """
        store = SessionStore(new_memory=lambda: None)
        s = store.get_or_create(None)
        store.commit(s, expected_version=0, question="q", answer="a")
        snap = s.messages()
        snap.append({"role": "user", "content": "偷偷加的"})
        assert len(s.history) == 2, "拿到的不是拷贝，改它会污染会话"

    def test_each_session_gets_its_own_lock(self):
        """一个会话一把锁，**不是全服务一把**。

        全服务一把的话，一个用户在算经济补偿，另一个用户的简单问答也得排队，
        并发度直接退化成 1。
        """
        store = SessionStore(new_memory=lambda: None)
        a = store.get_or_create("s-a")
        b = store.get_or_create("s-b")
        assert a.lock is not b.lock

    def test_lru_eviction_when_over_capacity(self):
        store = SessionStore(max_sessions=3, new_memory=lambda: None)
        for i in range(5):
            store.get_or_create(f"s-{i}")
        assert store.stats()["sessions"] == 3

    async def test_session_survives_across_requests(self):
        """两轮之间记忆对象是**同一个**，不是每轮重建。

        重建的话槽位表就永远只有当前这一轮的内容——
        表现为"它记不住我上一句说的月薪"，而那正是记忆层要证明的能力。
        """
        app, rt = _app([ANSWER] * 6)
        async with _client(app) as c:
            a = (await c.post("/api/chat", json={"question": "第一问"})).json()
            b = (
                await c.post("/api/chat", json={"question": "第二问", "session_id": a["session_id"]})
            ).json()
        assert a["session_id"] == b["session_id"]
        assert b["turn"] == 2


# --------------------------------------------------------------------------- #
# ⑤ 状态 → 响应的搬运动作（**这一组是回归测试，来自真机上踩到的坑**）
# --------------------------------------------------------------------------- #


class TestStateTransport:
    def test_values_state_is_a_dict_not_a_model(self):
        """钉死 langgraph 的一个**实测行为**：`values` 模式给的是 `dict`。

        为什么值得单写一条测试：这个事实决定了上层代码只能用
        "dict 和对象都能读"的方式访问状态（见 `api.app._g`）。
        如果哪天 langgraph 改成返回 `AgentState` 实例，`_g` 仍然能跑，
        **但这套认知就过期了**——到时候应该顺手把 `_g` 简化掉。
        **测试在这里的作用是"让这个假设变得可见"，而不是"防止它变"。**
        """
        run, *_ = _runner([ANSWER, ANSWER, ANSWER])
        last = None
        for mode, payload in run.stream("经济补偿怎么算", with_values=True):
            assert mode in {"updates", "values"}
            if mode == "values":
                last = payload
        assert last is not None, "with_values=True 没有产出 values 流"
        assert isinstance(last, dict), (
            f"values 模式不再是 dict 了（现在是 {type(last).__name__}）。"
            "请回头检查 api/app.py 的 _g()：它的存在前提就是这条。"
        )

    def test_values_state_may_omit_keys(self):
        """`values` 的 dict **可能整个缺键**（实测里 `refusal` 会缺席）。

        这条直接解释了为什么不能用 `payload["x"]`（KeyError）
        也不能用 `getattr(payload, "x", None)`（静默变 None）。
        """
        run, *_ = _runner([ANSWER, ANSWER, ANSWER])
        last = dict()
        for mode, payload in run.stream("经济补偿怎么算", with_values=True):
            if mode == "values":
                last = payload
        # `refusal` 是 `AgentState` 声明过的字段，但这一轮没有任何节点写过它，
        # 于是 `values` 给的 dict 里**干脆没有这个键**。
        assert "refusal" not in last, (
            "refusal 出现在 values 里了。如果 langgraph 的行为变了，"
            "请重读 api/app.py 的 _g() 并确认它还有必要。"
        )

    def test_g_reader_handles_dict_object_and_missing(self):
        """`_g` 对三种输入都要给出默认值而不是抛异常或变 None。"""
        from juris_agent.api.app import _g

        class Obj:
            answer = "对象里的答案"

        assert _g({"answer": "dict 里的答案"}, "answer", "") == "dict 里的答案"
        assert _g(Obj(), "answer", "") == "对象里的答案"
        assert _g({}, "answer", "") == ""
        assert _g(Obj(), "nope", "") == ""
        # ⚠️ 值为 None 时**返回默认值**而不是 None：
        # 后端把"没这个键"和"值是 None"当成同一件事处理。
        # 区分它们只会让每个调用点多写一个分支，而没有一个字段需要区分。
        assert _g({"answer": None}, "answer", "") == ""

    async def test_answer_is_not_silently_empty(self):
        """**最重要的一条回归测试。**

        起因：`values` 给的是 dict，而我一开始用 `getattr(state, "answer", "")` 读。
        `getattr` 读 dict **不报错**，它返回默认值 `""`。
        于是接口返回 `200 OK` + 一条空回答——没有异常、没有日志、
        没有红字，只能靠肉眼盯着屏幕发现。

        所以这里断言的是"答案的字真的到了响应里"，
        而不是"函数返回了 200"。**状态码证明不了任何事情。**
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()
        assert d["answer"].strip(), (
            "答案是空的。多半是状态被当成对象读了（它是 dict），"
            "或者 `values` 流根本没送到 `_to_response`。"
        )
        assert d["finish_reason"] == "answered"

    async def test_turn_usage_is_read_on_the_worker_thread(self):
        """**这一轮的成本必须带分项，不能走"按总 token 粗估"的退路。**

        这条测试是补一个真机上踩到的坑：`LLMClient.turn_usage` 读的是
        **线程本地**的账本，账是工作线程开的，而 `_cost_of` 跑在事件循环线程里。
        在那边读只能拿到空账，于是成本报表里 `prompt_tokens` 永远是 0、
        金额走退路，并附一句"未取到分项"。

        **它不报错，而且退路算出来的金额看着还挺合理**——所以第一版真机跑通时
        没人发现。我是盯着 `basis` 那一行想"为什么会走退路"才挖出来的。

        ⚠️ 这里的假件**刻意模仿线程本地这个行为**（记下开账的线程、
        别的线程来读就返回 0）。如果假件只是"`begin_turn` 之后就一直有值"，
        它就**测不出这个 bug**——假件比真件"好用"的地方，正是 bug 藏身的地方。
        """
        import threading

        from juris_agent.llm import Usage

        class ThreadLocalUsageLLM(ScriptedLLM):
            """行为与 `LLMClient` 一致的假件：账本是线程本地的。"""

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._owner: int | None = None
                self._usage = Usage()
                self.read_threads: list[int] = []

            def begin_turn(self) -> None:
                self._owner = threading.get_ident()
                self._usage = Usage()

            @property
            def turn_usage(self) -> Usage:
                self.read_threads.append(threading.get_ident())
                # 不是开账的那个线程 → 空账。**这就是 `threading.local()` 的行为。**
                if self._owner != threading.get_ident():
                    return Usage()
                return self._usage

            def chat(self, messages, **kw):
                resp = super().chat(messages, **kw)
                if self._owner == threading.get_ident():
                    self._usage = self._usage + Usage(
                        prompt_tokens=1200, completion_tokens=300, total_tokens=1500
                    )
                return resp

        s = _settings()
        r = FakeRetriever()
        llm = ThreadLocalUsageLLM([ANSWER, ANSWER, ANSWER])
        reg = build_registry(r)
        ex = ToolExecutor(reg, s)
        run = make_runner(retriever=r, registry=reg, executor=ex, llm=llm, settings=s)
        app, _ = _app([], runner=run, retriever=r, llm=llm, s=s)

        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()

        cost = d["cost"]
        assert cost is not None
        # 走退路的话 prompt_tokens 会是 0、basis 会写着"未取到分项"。
        assert cost["prompt_tokens"] > 0 or cost["completion_tokens"] > 0
        assert "未取到分项" not in (cost["basis"] or ""), (
            f"成本走了退路（basis={cost['basis']!r}）。多半是 usage 在"
            "事件循环线程里被读了，而账本在那个工作线程里。"
        )
        assert llm.read_threads, "从来没读过 turn_usage，说明 usage 根本没被取"

    async def test_trace_items_keep_their_fields(self):
        """嵌套元素（`TraceStep`）在 `values` 里**仍然是 pydantic 对象**。

        "外层是 dict、里层是对象"这个混合形状很容易假设错：
        假设"全变 dict 了"会在里层报 AttributeError，
        假设"全没变"会在外层静默给出空值。这条把两边的形状都钉住。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()
        assert d["trace"], "trace 是空的"
        first = d["trace"][0]
        assert first["node"], "trace 里的 node 丢了"
        assert isinstance(first["elapsed_ms"], (int, float))
        assert isinstance(first["detail"], dict)


# --------------------------------------------------------------------------- #
# ⑥ 成本字段的转运
# --------------------------------------------------------------------------- #


class TestCostTransport:
    """`CostBreakdown` → `CostOut` → JSON 这一段的字段完整性。

    **这一组测的是"字段有没有静默丢掉"，不是"钱算得对不对"。**
    钱怎么算在 `tests/test_cost.py` 里测；这里只管一件事：
    算出来的那个数有没有真的到对面。

    为什么这件事特别值得单独测：`CostOut` 是手工抄的一份镜像，
    pydantic 默认**忽略没声明的字段**。所以 `CostBreakdown` 上加了字段
    而 `CostOut` 忘了加，症状是"接口里那个字段凭空消失"——
    不报错、不打日志、没有测试会红。前端只会显示一个 0。
    """

    def test_cost_out_populates_every_field_it_declares(self):
        """**声明的每一个字段都要真的填上值。**

        ⚠️ 断言方式是"**拿一份每个字段都不是默认值的 CostBreakdown 转一遍，
        看有没有哪个字段还是默认值**"，而不是逐个字段写死期望值。
        这样以后往 `CostOut` 里加字段、却忘了在 `_cost_out` 里填，
        这条测试会**自动覆盖到新字段**——不用回头改测试。

        `total_tokens` 就是这么被抓出来的：它在契约里躺了很久，
        每次都返回 `0`，而 `CostBreakdown` 里根本没有这个字段，
        `model_dump()` 里没有这个键，pydantic 就拿了默认值。
        前端没用它，所以一直没人发现。
        """
        from juris_agent.api.app import _cost_out
        from juris_agent.api.schemas import CostOut
        from juris_agent.observability import CostBreakdown

        src = CostBreakdown(
            model="deepseek-flash",
            prompt_tokens=14352,
            cache_hit_tokens=5504,
            cache_miss_tokens=8848,
            completion_tokens=2903,
            reasoning_tokens=1533,
            input_cny=0.0123,
            output_cny=0.0232,
            total_cny=0.0355,
            as_of="2026-10",
            basis="测试用，别当真",
            # 故意取**非默认值** False：默认也是 False 的话就分不出
            # "传过去了"和"没传、拿了默认值"。
            billable=False,
        )
        out = _cost_out(src)
        assert out is not None

        for name, field in CostOut.model_fields.items():
            if name == "total_tokens":
                continue  # 这个字段 `CostBreakdown` 里没有，单独断言
            expected = getattr(src, name)
            assert getattr(out, name) == expected, (
                f"`CostOut.{name}` 没被填上（={getattr(out, name)!r}，应为 {expected!r}）。"
                "多半是 `_cost_out` 里漏了这个字段，或者 `CostBreakdown` 改了名。"
            )

    def test_total_tokens_is_populated(self):
        """`total_tokens` 必须是"入 + 出"，**不能永远是 0**。

        这个字段在契约里存在很久了，但一直是死的（见上面那条测试的说明）。
        前端不用它、接口不报错，所以它静静地骗了很久。
        """
        from juris_agent.api.app import _cost_out
        from juris_agent.observability import CostBreakdown

        out = _cost_out(CostBreakdown(prompt_tokens=14352, completion_tokens=2903))
        assert out is not None
        assert out.total_tokens == 17255

    async def test_cost_reaches_json_with_billable(self):
        """成本必须带 `billable` 到 JSON 里，**前端全靠它区分"花了"和"重放"**。

        前端和演示脚本的判断顺序是"先看 `billable` 再看金额"。
        这个字段没到前端的话，`replay` 模式会继续显示 `¥0.138040`——
        一个一分钱没花的页面。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()
        assert "billable" in d["cost"], "CostOut 里没有 billable，前端拿不到"
        assert d["cost"]["total_tokens"] == (
            d["cost"]["prompt_tokens"] + d["cost"]["completion_tokens"]
        )

    async def test_a_turn_with_no_recorded_call_is_not_billable(self):
        """**没记账的一轮不能报成花过钱——这是真机上踩过的坑。**

        `ScriptedLLM` 没有 `begin_turn`，所以它和陈旧缓存走的是**同一条路**：
        线程本地的账是空的 → `_cost_of` 走退路 → 金额按状态里的
        `tokens_used` 粗估。那个 `tokens_used` 是**真的**（假模型每次调用
        都记 10 个 token），于是能算出一个**非 0** 的金额。

        这正是 `replay` 模式的样子：数是真的，钱不是这次花的。
        所以这里同时断言两件事——**金额不为 0**（对照要用的数还在），
        **`billable` 为 False**（但这一轮没花钱）。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER])
        async with _client(app) as c:
            d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()

        cost = d["cost"]
        assert cost["prompt_tokens"] + cost["completion_tokens"] > 0, (
            "token 数是 0 的话，这条测试就没测到'有数但不计费'这个形状"
        )
        assert cost["billable"] is False, (
            "这一轮一次计费调用都没发生，却标成了'花过钱'——"
            " replay 模式就是这么把 0.138 元报出来的"
        )


class TestRequestId:
    """**用户看到的请求 id 必须能在日志里搜到。**（F28 的验收标准）

    这条测试防的是一个真 bug：`_run_turn` 里生成一个 id 塞进 `done` 事件，
    调用方又生成**另一个** id 去 `bind()` 日志上下文。
    两个串格式一样、都正常生成，但不是一个东西。

    症状：界面上显示"请求 abc123"，用户把这串字发过来排查，
    `grep abc123` **一条都搜不到**——而日志里确实有一批同格式的 id 在跑，
    于是看起来像"这个人抄错了"或者"日志没开"。**排查方向被带偏了。**
    """

    async def test_done_event_id_matches_the_id_bound_to_logs(self):
        """`done` 事件里的 id 和 `bind()` 到日志上下文里的 id 必须是同一个。

        ⚠️ **断言的是"这两处用了同一个串"，而不是"日志里出现过这个串"。**
        一开始我写的是后者——用一个假 logger 截获"这一轮日志里的
        `request_id`"。结果它**什么都没截到**：正常一轮问答**一条日志都不打**
        （日志只在出错、会话写回冲突、客户端断开时才有）。
        那种写法看起来更"端到端"，但它测的是一条根本不存在的日志，
        所以永远会红，而且红的原因和被测的 bug 没关系。

        **测试要盯着那个真的会出错的地方**：这里会出错的只有一件事——
        两个地方各 `new_request_id()` 了一次。所以就盯这个。
        """
        import juris_agent.api.app as app_mod

        bound: list[str] = []
        real_bind = app_mod.bind

        def spy_bind(*a, **kw):
            # 记下"绑进日志上下文的那个 id"，然后照常绑。
            if kw.get("request_id"):
                bound.append(kw["request_id"])
            return real_bind(*a, **kw)

        app_mod.bind = spy_bind
        try:
            app, _ = _app([ANSWER, ANSWER, ANSWER, ANSWER, ANSWER, ANSWER])
            async with _client(app) as c:
                # **两条路都要查。** 网页走的是 SSE 那条，
                # 所以用户真正会遇到的其实是第二条——只测非流式那条，
                # 等于把 bug 留在用户唯一会用的路径上。
                d1 = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()
                text = (
                    await c.post(
                        "/api/chat/stream",
                        json={"question": "经济补偿怎么算", "stream_llm": True},
                    )
                ).text
        finally:
            app_mod.bind = real_bind

        frames = _frames(text)
        done = next(data for name, data in frames if name == "done")

        for label, payload in (("非流式", d1), ("SSE", done)):
            shown = payload.get("request_id", "")
            assert shown, f"{label}的响应里没有 request_id"
            assert shown in bound, (
                f"{label}：界面上的请求 id（{shown!r}）和日志上下文里的"
                f"（{bound}）不是同一个。症状：用户把这串字发过来排查，"
                "grep 一条都搜不到。多半是 `_run_turn` 和调用方各生成了一次 id。"
            )

    async def test_request_id_is_actually_greppable_in_the_logs(self, capsys):
        """**拿界面上的 id 去日志里搜，要真能搜到那一轮。**（F28 的验收原文）

        上一条测的是"两处用了同一个串"，是**机制**；这一条测的是**结果**：
        把日志抓下来，看那个 id 在不在里面。

        为什么要单独立一条：最早写的时候，正常一轮问答**一条日志都不打**
        （日志只在出错、会话写回冲突、客户端断开时才出现）。
        于是 `request_id` 在实际使用中的唯一效果是"界面上多了一串字符"——
        用户说"我这轮回答很怪"，你拿着那串 id 去搜，**什么都搜不到**。
        **一个只在出事时才有内容的排查机制，在最需要它的
        "说不出哪里不对"的场景里是哑的。** 现在每轮结束会留一行。

        ⚠️ 这里用 `capsys` 抓**真实输出的日志**，不是去打桩 logger。
        打桩的话，"日志到底有没有打出去、字段有没有被渲染进去"
        这两件事都测不到——而它们恰好是最容易坏的。
        """
        import json as _json

        from juris_agent.observability import configure_logging

        # 用 JSON 渲染器抓：一行一条，能直接 json.loads。
        configure_logging(level="INFO", json=True)
        try:
            app, _ = _app([ANSWER, ANSWER, ANSWER])
            async with _client(app) as c:
                d = (await c.post("/api/chat", json={"question": "经济补偿怎么算"})).json()
        finally:
            configure_logging(level="INFO", json=False)

        shown = d["request_id"]
        lines = []
        for ln in capsys.readouterr().out.splitlines():
            ln = ln.strip()
            if ln.startswith("{"):
                with contextlib.suppress(_json.JSONDecodeError):
                    lines.append(_json.loads(ln))

        matched = [x for x in lines if x.get("request_id") == shown]
        assert matched, (
            f"界面上的请求 id（{shown!r}）在日志里一条都搜不到。"
            f"这一轮打出来的日志是：{[x.get('event') for x in lines]}"
        )
        # 光"搜到了"还不够：那一行得**带着能判断问题的信息**，
        # 否则它只是一条"有个请求发生过"的废话。
        turn_log = next(x for x in matched if x.get("event") == "一轮结束")
        for field in ("route", "finish", "tokens", "session_id"):
            assert field in turn_log, f"这一行日志缺 {field}，排查时还是得去别处找：{turn_log}"

        # ⚠️ **"那一轮"的日志也要算上建会话那一条。**
        #
        # 这条断言是后补的：一开始 `bind(request_id=...)` 套在
        # `get_or_create()` **里面**，而 `session_id` 必须等建完才有，
        # 于是 "新建会话" 成了整个请求里**唯一一条没有 request_id 的日志**。
        # 后果是把同一个请求的日志劈成两半：一半能按 id 搜到，
        # 另一半（恰恰是"这会话是什么时候建的"这条）只能靠时间戳猜。
        # **能按 id 串起来才算"串起来"**——有一条漏网，这个承诺就是半真的。
        new_session_log = next(
            (x for x in lines if x.get("event") == "新建会话"), None
        )
        assert new_session_log is not None, (
            "这一轮没有「新建会话」的日志，这条断言测不到东西了。"
            f"打出来的日志是：{[x.get('event') for x in lines]}"
        )
        assert new_session_log.get("request_id") == shown, (
            "同一个请求的日志被劈成了两半：「新建会话」那条的 request_id 是 "
            f"{new_session_log.get('request_id')!r}，而这一轮的 id 是 {shown!r}。"
            "多半是把 `bind(request_id=...)` 套在了 `get_or_create()` 里面，"
            "于是建会话时上下文里还没有 id。"
        )


# --------------------------------------------------------------------------- #
# ⑦ 法条原文接口
# --------------------------------------------------------------------------- #


class _RetrieverWithArticles(FakeRetriever):
    """给 `articles_by_key` 补上，测 `/api/article/{key}` 用。

    ⚠️ 假件和真件在这里**刻意不一样**：真检索器有这个属性，假件没有。
    这个差异本身有价值——它提醒"接口依赖了假件没模拟的属性"，
    而那种接口在测试里全绿、一上真环境就 503。
    """

    def __init__(self):
        super().__init__()
        self.articles_by_key = {"lcl#47": _FakeArticle()}


class _FakeArticle:
    article_key = "lcl#47"
    law_name = "中华人民共和国劳动合同法"
    law_short = "劳动合同法"
    article_no = "第四十七条"
    chapter = "第七章 法律责任"
    section = None
    text = "经济补偿按劳动者在本单位工作的年限……"
    effective_date = None
    version_date = None
    effective_date_source = "manifest"

    def citation(self) -> str:
        return "《劳动合同法》第四十七条"


class TestArticle:
    async def test_returns_text_and_both_law_names(self):
        """必须同时给全称和简称。

        简称是**引用时需要和答案里编号一致**的那个；
        全称是给人看、给外行确认"这是哪部法"的那个。只给一个都会有人不满意。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER], retriever=_RetrieverWithArticles())
        async with _client(app) as c:
            r = await c.get("/api/article/lcl%2347")
        assert r.status_code == 200
        d = r.json()
        assert d["citation"] == "《劳动合同法》第四十七条"
        assert d["law_name"] == "中华人民共和国劳动合同法"
        assert d["law_short"] == "劳动合同法"
        assert d["text"]

    async def test_unknown_key_is_404(self):
        app, _ = _app([ANSWER, ANSWER, ANSWER], retriever=_RetrieverWithArticles())
        async with _client(app) as c:
            r = await c.get("/api/article/lcl%2399")
        assert r.status_code == 404

    async def test_missing_index_is_503_not_404(self):
        """**检索器没装配是 503，不是 404。**

        区别很实在：404 等于告诉调用方"这条法条不存在"，
        而事实是"我现在查不了"。混用会让排查方向完全跑偏。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER], retriever=FakeRetriever())
        async with _client(app) as c:
            r = await c.get("/api/article/lcl%2347")
        assert r.status_code == 503


# --------------------------------------------------------------------------- #
# ⑦ 静态页面
# --------------------------------------------------------------------------- #


class TestWeb:
    async def test_index_is_served_when_enabled(self):
        app, _ = _app([ANSWER, ANSWER, ANSWER], serve_web=True)
        async with _client(app) as c:
            r = await c.get("/")
        assert r.status_code == 200
        body = r.text
        # 页面必须真的是能对话的界面，不是一张说明页。
        # 这几条断言就是"产品壳"的验收：有输入框、接了流式接口、有法条区。
        assert "/api/chat/stream" in body, "页面没有接流式接口"
        assert "<textarea" in body, "页面没有输入框"
        assert "不是法律意见" in body, "免责声明没了"

    async def test_page_renders_the_calculation_steps(self):
        """页面要真的把计算题的**算式与代入过程**画出来。

        ⚠️ 这条是**字符串检查**，很弱，它证明不了渲染逻辑是对的。
        留着它是因为它防的是最可能发生的那种坏法：**有人把这段删了**。
        那件事的症状是"计算题的答案只有一个结论"，
        而**后端的测试全绿**（工具照样在算 steps、事件照样在传 payload——
        见 `TestStream::test_tool_result_carries_the_calculation_steps`）。
        两半分开测，因为它们是两处不同的代码、会各自坏掉。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER], serve_web=True)
        async with _client(app) as c:
            body = (await c.get("/")).text
        assert "payload" in body, "页面根本没读 tool_result 的 payload"
        assert ".steps" in body, "页面没有取 payload.steps"
        assert "算式" in body, "页面上没有'算式'这层展示"

    async def test_index_absent_when_disabled(self):
        app, _ = _app([ANSWER, ANSWER, ANSWER], serve_web=False)
        async with _client(app) as c:
            r = await c.get("/")
        assert r.status_code == 404

    async def test_page_has_no_external_requests(self):
        """页面**不许有外部请求**（CDN / 字体 / 图标）。

        这条是拿"演示能不能打开"换来的：现场网络差、CDN 被墙，
        一个引了外部脚本的页面打不开的概率不为零，而那一刻你没法解释。
        """
        app, _ = _app([ANSWER, ANSWER, ANSWER], serve_web=True)
        async with _client(app) as c:
            body = (await c.get("/")).text
        for bad in ("http://", "https://", "//cdn", "unpkg", "jsdelivr"):
            # 允许出现 `http://test`（我们自己在测试里用的），
            # 但正文里不该有任何真的外链。
            assert bad not in body.replace("http://test", ""), f"页面里有外链：{bad}"


# --------------------------------------------------------------------------- #
# ⑧ 装配：允许降级，但不许悄悄降级
# --------------------------------------------------------------------------- #


class TestRuntime:
    def test_degraded_but_constructible(self):
        """缺依赖时**照样构造出 Runtime**，把问题记在 `problems` 里。

        反过来的做法（缺东西就抛异常）会让演示现场"只想看一眼前端长什么样"
        也必须先装完 2.5GB 的模型。
        """
        rt = Runtime(settings=_settings(), sessions=SessionStore(new_memory=lambda: None))
        assert rt.ready is False
        assert rt.health()["ok"] is True, "服务本身是好的，只是答不了问题"

    def test_close_is_safe_without_llm(self):
        """没有 LLM（或 LLM 没有 close）时 `close()` 不能抛。

        它会在 lifespan 的 finally 里被调用，**抛出去会掩盖真正的启动异常**。
        """
        rt = Runtime(settings=_settings(), sessions=SessionStore(new_memory=lambda: None))
        rt.close()
