"""LLM 调用层：一个很小的 DeepSeek 客户端 + **内容寻址缓存**。

这个模块有两个刻意的设计，都不是"能跑就行"，而是被具体的问题逼出来的。

## 一、为什么不用 `openai` SDK，直接用 `httpx`

DeepSeek 的 API 是 OpenAI 兼容的，所以 `openai` 包确实能用。不用的理由：

1. **重试语义会打架。** `openai` 客户端自带 `max_retries`，而本项目在工具层
   （`tools/executor.py`）已经有自己的退避重试与熔断。两层重试叠在一起 = 最坏情况下
   `3 × 3 = 9` 次真实请求，故障注入实验里那条第 4 项结论（"重试单独用是有害的，
   它让已经过载的下游承受 3 倍流量"）会被放大成 9 倍，而且**日志里看不出来**——
   你只能看到一个工具失败了，不知道底下打了 9 发。
2. **我们本来就要自己写缓存层。** 请求体是二十来行的 JSON，自己发反而更透明。
3. 少一个会跟着上游改 API 面的依赖。

## 二、为什么要缓存（而且键必须是"内容"）

**这不是为了省钱，是为了让消融实验可比。**

考虑 F9 查询改写那一臂：它用 LLM 把「老板拖欠工资怎么办」改写成法条用语。
如果我们跑两次消融、改写结果不同，那么"加了改写之后 Recall 涨了 3.2 个点"
里到底有多少来自改写、有多少来自这次恰巧改写得好——**分不出来**。

所以缓存键 = `sha256(model + messages + temperature + max_tokens + ...)`，
即"同样的输入必然拿到同样的输出"。三种模式：

| 模式 | 读缓存 | 未命中时 | 用途 |
|---|---|---|---|
| `off` | 否 | 调 API，不写缓存 | 日常开发 |
| `record` | 是 | 调 API 并写缓存 | 第一次跑、补充新样本 |
| `replay` | 是 | **抛错** | 复现实验、离线调参，零成本零等待 |

`replay` 未命中直接抛错是**故意**的：静默回落到真实 API 会让"离线复现"这个承诺失效，
而你可能过很久才发现自己在花真钱。

产出落在 `fixtures/replay/llm/*.json`（`.gitignore` 里已排除——它可能含 prompt 全文）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from juris_agent.config import get_settings

# --------------------------------------------------------------------------- #
# 错误分类
# --------------------------------------------------------------------------- #


class LLMError(RuntimeError):
    """LLM 调用失败。可重试与不可重试靠 `retryable` 区分。

    与 `tools/errors.py` 的分类原则一致：**4xx 不重试**——参数错、鉴权错、
    额度耗尽，重试只会把同样的错误再犯一遍，还顺便把额度烧掉。
    """

    retryable: bool = False

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LLMRateLimit(LLMError):
    retryable = True


class LLMServerError(LLMError):
    retryable = True


class LLMTimeout(LLMError):
    retryable = True


class LLMAuthError(LLMError):
    retryable = False


class LLMBadRequest(LLMError):
    retryable = False


class ReplayMiss(LLMError):
    """`replay` 模式下缓存未命中。不重试、不回落——见模块 docstring。"""


class LLMTruncated(LLMError):
    """**推理模型的专属坑**：思考过程也计入 `max_tokens`。

    实测：`deepseek-flash` 在 `max_tokens=32` 时 `content` 是**空字符串**，
    而 `completion_tokens` 正好是 32——32 个 token 全花在思维链上，
    正文一个字都没轮到。

    危险之处在于它**不报错**：HTTP 200、字段齐全、解析得动，
    只是内容为空。如果下游（比如 `complete_json`）拿到空串就去 `json.loads`，
    报的会是"JSON 解析失败"，于是人往"模型不听话"的方向查，
    而真正的原因是"预算给小了"。

    处理方式见 `chat()`：自动把预算翻几倍重试，而不是把空串交给调用方。
    """


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # DeepSeek 分"缓存命中/未命中"两种计价，命中部分便宜一个数量级。
    # 记账时要分开记，否则成本报表会高估。
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        # 用 `type(self).model_fields` 而不是 `self.model_fields`——
        # 后者在 pydantic 2.11 起已废弃（PydanticDeprecatedSince211），
        # 且在本套件里会刷出 21 条 warning 把真正的告警淹掉。
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in type(self).model_fields})


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    from_cache: bool = False

    reasoning_text: str = ""
    """推理模型的思考过程（流式时才有值）。

    ⚠️ **它已经计在 `usage.completion_tokens` 里**（供应商是这么算的），
    所以**成本记账绝不能把它再加一遍**。它存在的唯一理由是
    "答案很短但很贵，钱花哪了"这个问题——前端把它显示成"思考中"，
    也是为了回答同一个问题。默认空串：非流式调用拿不到它，
    老缓存文件里也没有这个字段，空串让两者都能正常读出来。"""

    # Function Calling（工具调用）。模型"想调工具"时，`text` 通常是空串，
    # 内容全在这里。**必须进缓存**，否则 replay 时工具调用会凭空消失——
    # 那种复现出来的轨迹是假的，比不复现更危险。
    #
    # 结构照抄 OpenAI 协议（DeepSeek 兼容）：
    # `[{"id": "call_1", "type": "function",
    #    "function": {"name": "calc_severance", "arguments": "{\"months_of_service\": 36}"}}]`
    #
    # ⚠️ `arguments` 是**字符串**不是对象（协议就这么定的，服务端不解析它）。
    # 所以拿到的第一件事是 `json.loads`，而它**可能解析失败**——
    # 模型偶尔会输出带注释或尾逗号的 JSON。这一层不做兜底，
    # 由 `agent/nodes.py` 把解析错误变成一条"参数不合法"的工具反馈喂回模型让它重试，
    # **那比在这里静默修 JSON 好**：修错了会执行一个用户没要求的计算。
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- #
# 内容寻址缓存
# --------------------------------------------------------------------------- #


def _classify_http(status: int, text: str) -> LLMError:
    """把 HTTP 状态码翻成一个错误对象。**`_post` 和 `_post_stream` 共用这一份。**

    为什么必须共用：流式和非流式是**同一个接口的两种传输方式**，
    对同一个状态码给出不同的处理（比如流式那条路把 429 当成永久错误）
    会让"重试不重试"取决于"用户点没点流式开关"——这种 bug 极难发现，
    因为它只在你打开流式的时候才出现。

    返回错误对象而不是抛，是因为两条路径的**处理方式不同**：
    非流式可以重试整个请求；流式一旦已经把前半段 token 交给用户了，
    就没法"重新来过"了（会把前面的字重复打一遍）。
    """
    if status in (401, 403):
        return LLMAuthError(f"鉴权失败 HTTP {status}: {text[:200]}", status=status)
    if status == 429:
        return LLMRateLimit(f"限流 HTTP 429: {text[:200]}", status=429)
    if 400 <= status < 500:
        return LLMBadRequest(f"请求错误 HTTP {status}: {text[:300]}", status=status)
    return LLMServerError(f"服务端错误 HTTP {status}: {text[:200]}", status=status)


def _cache_key(payload: dict[str, Any]) -> str:
    """键只由**影响输出的入参**决定；`stream` 之类不影响结果的不进键。

    ⚠️ `tools` / `tool_choice` **必须进键**。漏了会导致：
    "不带工具的一次调用"和"带工具的一次调用"算出同一个 key，
    于是 replay 时读到的是**另一次调用的结果**——工具调用凭空出现或消失，
    指标看着正常，轨迹是假的。第 33 条需求（record-replay）里
    "改了 prompt 会自然 miss"这句话，对 tools 同样成立。
    """
    material = {
        "model": payload.get("model"),
        "messages": payload.get("messages"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "top_p": payload.get("top_p"),
        "response_format": payload.get("response_format"),
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
    }
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class LLMCache:
    """一个目录一层 = 一次缓存。线程安全（gold 生成要并发）。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        p = self.path_for(key)
        if not p.exists():
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        resp = LLMResponse(**json.loads(p.read_text(encoding="utf-8")))
        resp.from_cache = True
        resp.latency_ms = 0.0
        return resp

    def put(self, key: str, resp: LLMResponse) -> None:
        with self._lock:
            self.path_for(key).write_text(
                resp.model_dump_json(indent=1), encoding="utf-8"
            )

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #


class LLMClient:
    """DeepSeek chat/completions 客户端。

    并发安全：`complete_many` 用线程池；每个线程复用同一个 `httpx.Client`
    （它是线程安全的，比每线程建一个连接池更省）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        mode: Literal["off", "record", "replay"] | None = None,
        cache_dir: Path | None = None,
        client: httpx.Client | None = None,
        max_retries: int | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = api_key or s.deepseek_api_key
        if not self.api_key:
            raise LLMAuthError(
                "没有 DEEPSEEK_API_KEY。复制 .env.example 为 .env 并填入 key（见 README「安装」）。"
            )
        self.base_url = (base_url or s.deepseek_base_url).rstrip("/")
        self.mode: Literal["off", "record", "replay"] = mode or s.llm_mode  # type: ignore[assignment]
        self.model_gen = s.model_gen
        self.model_judge = s.model_judge
        self.max_retries = s.llm_max_retries if max_retries is None else max_retries
        self.cache = LLMCache(cache_dir or (s.paths.fixtures / "replay" / "llm"))
        self._client = client or httpx.Client(timeout=s.llm_timeout_seconds)
        self._owns_client = client is None
        # 记账：整个进程累计，脚本结束时打出来
        self.usage_total = Usage()
        self._usage_lock = threading.Lock()
        # 按**线程**记的另一本账，给"这一次请求花了多少"用。见 `begin_turn`。
        self._turn = threading.local()

    # -- 记账：整进程 vs 单次请求 ------------------------------------------ #

    def begin_turn(self) -> None:
        """在**当前线程**里开一本新账。

        ## 为什么不能靠"调用前后各读一次 `usage_total` 做差"

        那是第一版的做法，并发下**直接算错**：两个请求重叠时，
        A 做差得到的数是"A 加上 B 中途花掉的部分"。而且它**不会报错**——
        成本报表上只会看到一个偏大的数，没有任何人看得出来。

        ## 线程本地为什么在这里是够用的

        本服务每次请求的整轮处理都跑在**同一个工作线程**里
        （`api/app.py` 用 `asyncio.to_thread` 把同步的图丢给线程池）。
        所以"线程"恰好和"一次请求"对齐，线程本地就等价于请求本地。

        ⚠️ **这个前提是必须说清楚的**：如果哪天改成"一次请求跨多个线程"
        （比如工具并发执行），这本账就会把几个线程的数字混起来。
        那时要做的是换成显式传入的记账对象，而不是在这里打补丁。

        ⚠️ 线程是被线程池**复用**的，所以必须在每轮开始时重置——
        不重置的话这一轮会带着上一轮的账。
        """
        self._turn.usage = Usage()

    @property
    def turn_usage(self) -> Usage:
        """当前线程自 `begin_turn()` 以来真实花掉的 token。

        **没调过 `begin_turn()` 时返回全 0**——脚本、单测里没人会去调它。
        返回 0 而不抛异常，是因为"这次没记账"和"这次没花钱"
        对调用方的正确行为一样（都是别去读它），而抛异常会让
        `chat()` 里多加一个 try。
        """
        return getattr(self._turn, "usage", Usage())

    def _record_usage(self, usage: Usage) -> None:
        """两本账一起记。**只在这一个地方记**，免得漏了一本。"""
        with self._usage_lock:
            self.usage_total = self.usage_total + usage
        if hasattr(self._turn, "usage"):
            self._turn.usage = self._turn.usage + usage

    # -- 生命周期 ---------------------------------------------------------- #

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 核心 -------------------------------------------------------------- #

    def _post(self, payload: dict[str, Any]) -> LLMResponse:
        """发一次请求，带本层自己的退避重试。**只重试 `retryable` 的错。**"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload
                )
            except httpx.TimeoutException as e:
                last = LLMTimeout(f"请求超时: {e}")
            except httpx.HTTPError as e:
                last = LLMServerError(f"网络错误: {e}")
            else:
                if r.status_code == 200:
                    body = r.json()
                    msg = body["choices"][0]["message"]
                    u = body.get("usage", {}) or {}
                    details = u.get("completion_tokens_details", {}) or {}
                    text = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls") or []
                    completion = u.get("completion_tokens", 0)
                    # 空正文 + 消耗了 token = 预算被思维链吃光了（见 LLMTruncated）。
                    # 抛出去让 `chat()` 加预算重试，绝不把空串往下游传。
                    #
                    # ⚠️ `and not tool_calls` 这个条件是**必需的**：模型决定调工具时
                    # 正文本来就是空的（内容在 tool_calls 里），token 也确实花了。
                    # 少了这个判断，每次工具调用都会被误判成"预算不够"，
                    # 于是自动翻 4 倍重试——**白烧一遍钱，而且重试结果被缓存下来，
                    # 后面所有人都吃到这个被污染的结果**。
                    if not text.strip() and not tool_calls and completion > 0:
                        raise LLMTruncated(
                            f"模型返回空正文（completion_tokens={completion}，"
                            f"reasoning_tokens={details.get('reasoning_tokens', 0)}），"
                            f"max_tokens={payload.get('max_tokens')} 不够推理模型用"
                        )
                    return LLMResponse(
                        text=text,
                        model=body.get("model", payload.get("model", "")),
                        usage=Usage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=completion,
                            total_tokens=u.get("total_tokens", 0),
                            prompt_cache_hit_tokens=u.get("prompt_cache_hit_tokens", 0),
                            prompt_cache_miss_tokens=u.get("prompt_cache_miss_tokens", 0),
                            reasoning_tokens=details.get("reasoning_tokens", 0),
                        ),
                        tool_calls=tool_calls,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                if r.status_code in (401, 403):
                    raise _classify_http(r.status_code, r.text)
                if r.status_code == 429:
                    last = _classify_http(r.status_code, r.text)
                elif 400 <= r.status_code < 500:
                    # 4xx 不重试。抛出去让调用方看见真实的错误原因。
                    raise _classify_http(r.status_code, r.text)
                else:
                    last = _classify_http(r.status_code, r.text)

            if attempt < self.max_retries and last is not None and last.retryable:
                time.sleep(min(2.0 ** attempt, 8.0))
                continue
            break
        raise last or LLMError("未知失败")

    def _post_stream(
        self,
        payload: dict[str, Any],
        on_delta: Any,
        on_reasoning: Any,
    ) -> LLMResponse:
        """流式发一次请求：边收边把增量交给回调，最后拼成和 `_post` 一样的返回值。

        ## 为什么不另开一个 `stream()` 方法

        如果另开一个方法，`chat()` 里那些**必须做的事**——缓存读写、
        额度不够时自动升级重试、usage 累计——就得在两边各写一遍。
        **缓存那条尤其危险**：漏了它，流式路径的响应不进缓存，
        于是"离线可复现"这个承诺在开了流式的场景下悄悄失效，
        而且失效的方式是"replay 时查不到 → 报错"，看起来像是录少了，
        根本想不到是"用了另一条代码路径"。

        所以这里把它做成 `_post` 的**传输层替身**：上层 `chat()` 一行不改。

        ## 两条硬约束

        **① 一旦吐出去一个字，就不能再重试了。** 重试会把同一段话
        从头再吐一遍，用户看到的是"试用期最长……试用期最长不得超过六个月"。
        所以重试只在 `emitted == 0` 时进行——这个条件比"收到了多少字节"
        更可靠，因为**没吐出去的字用户根本没看见**。

        **② `stream_options.include_usage` 必须带上。** 不带的话，
        流式响应里**没有 usage 字段**，成本记账会全部记成 0，
        而 0 和"没算"在报表上长得一模一样。带上它之后，
        服务端会在最后一个 chunk 里补一份 usage。
        """
        stream_payload = {
            **payload,
            "stream": True,
            # 这一步是"要 usage"的开关。它不改变答案，只影响最后一个 chunk
            # 里多不多一个 usage 字段。所以**它不进缓存键**——
            # `_cache_key` 只取那几个影响输出的字段，加不加它算出来是同一个 key。
            # 这是刻意的：同一次调用在流式和非流式下应当读到同一份缓存。
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            usage = Usage()
            emitted = 0

            try:
                with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=stream_payload,
                ) as r:
                    if r.status_code != 200:
                        # 错误响应体是普通 JSON，不是事件流。先读完再分类。
                        r.read()
                        err = _classify_http(r.status_code, r.text)
                        if not err.retryable:
                            raise err
                        last = err
                        continue

                    for raw in r.iter_lines():
                        line = raw.strip()
                        # SSE 的空行是分隔符，`:` 开头的是心跳注释——都不是数据。
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            # 单块解析不了就跳过。**不抛**——一块坏数据
                            # 不该让整段已经收到的答案作废。
                            continue

                        if u := chunk.get("usage"):
                            details = u.get("completion_tokens_details", {}) or {}
                            usage = Usage(
                                prompt_tokens=u.get("prompt_tokens", 0),
                                completion_tokens=u.get("completion_tokens", 0),
                                total_tokens=u.get("total_tokens", 0),
                                prompt_cache_hit_tokens=u.get("prompt_cache_hit_tokens", 0),
                                prompt_cache_miss_tokens=u.get("prompt_cache_miss_tokens", 0),
                                reasoning_tokens=details.get("reasoning_tokens", 0),
                            )

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}

                        if piece := delta.get("content"):
                            text_parts.append(piece)
                            emitted += 1
                            if on_delta is not None:
                                on_delta(piece)
                        if piece := delta.get("reasoning_content"):
                            reasoning_parts.append(piece)
                            if on_reasoning is not None:
                                on_reasoning(piece)

                        # 流式下 tool_calls 是**碎片**：第一块给 id 和函数名，
                        # 后面几块只给参数字符串的一小段，靠 `index` 拼起来。
                        # 拼错的话结果是参数 JSON 不完整——下游 `json.loads` 会失败，
                        # 表现为"模型老是填不对参数"，而真相是我们没拼好。
                        for frag in delta.get("tool_calls") or []:
                            idx = int(frag.get("index", 0))
                            slot = tool_calls.setdefault(
                                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            )
                            if fid := frag.get("id"):
                                slot["id"] = fid
                            fn = frag.get("function") or {}
                            if fname := fn.get("name"):
                                slot["function"]["name"] = fname
                            if fargs := fn.get("arguments"):
                                slot["function"]["arguments"] += fargs

            except httpx.TimeoutException as e:
                last = LLMTimeout(f"请求超时: {e}")
                # ⚠️ 已经吐出去一个字就不能重试（见 docstring 约束 ①）：
                # 重试会让用户看到同一段话打两遍。这里直接放弃并把错误抛出去，
                # 让调用方决定怎么收场——**半段答案通常比"系统忙"更有用**。
                if emitted:
                    break
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
                break
            except httpx.HTTPError as e:
                last = LLMServerError(f"网络错误: {e}")
                if emitted:
                    break
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
                break
            else:
                text = "".join(text_parts)
                calls = [tool_calls[i] for i in sorted(tool_calls)]
                completion = usage.completion_tokens
                # 和非流式路径**同一条判据**：空正文 + 没工具调用 + 花了 token
                # = 预算被思维链吃光了（见 `LLMTruncated`）。
                #
                # 抛出去之后由 `chat()` 接住并翻 4 倍预算重来。
                # 这里比非流式多一层隐式保证：**走到这一行就说明一个字都没吐出去**
                # （有正文就不会进这个分支），所以"重来一遍"不会让用户看到重复内容。
                if not text.strip() and not calls and completion > 0:
                    raise LLMTruncated(
                        f"模型返回空正文（completion_tokens={completion}，"
                        f"reasoning_tokens={usage.reasoning_tokens}），"
                        f"max_tokens={payload.get('max_tokens')} 不够推理模型用"
                    )
                return LLMResponse(
                    text=text,
                    model=payload.get("model", ""),
                    usage=usage,
                    tool_calls=calls,
                    reasoning_text="".join(reasoning_parts),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

            # 走到这里只有两种可能：上面 `else` 分支 return 了（不会到这），
            # 或者某个 except 分支 `break` 了（已经出循环）。
            # 所以循环体末尾**没有**收尾代码——重试的决策全部写在
            # "为什么失败"的那一处，不在这里再判一遍（判两遍就会不一致）。
        raise last or LLMError("未知失败")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Any = None,
        on_reasoning: Any = None,
    ) -> LLMResponse:
        """一次对话补全。

        `temperature` 默认 0：**评测路径上必须是 0**，否则同样输入两次跑出两个数，
        消融就不可比。需要采样多样性的地方（如 F26 的"同一题问 3 次取多数"）
        由调用方显式传 temperature > 0，那时它自己负责去聚合。

        ⚠️ **`temperature=0` 不等于可复现**——本项目实测过：同一道题连问两次，
        改写结果 20/20 不同。托管推理模型的解码在这个层面就不确定。
        它买到的是"稳定一些"，不是"确定性"。要真正的可复现只有 `replay` 模式。

        `messages` 的类型放宽到 `dict[str, Any]` 是为了 Function Calling：
        工具调用往返里会出现 `{"role": "assistant", "content": null, "tool_calls": [...]}`
        和 `{"role": "tool", "tool_call_id": "call_1", "content": "{...}"}` 这两种消息，
        它们的字段不是纯字符串。

        ## 流式：传 `on_delta` 就开

        `on_delta(piece: str)` 会被**逐段**调用，`on_reasoning(piece)` 收到的是
        推理模型的思考过程（`deepseek-flash` 这类模型会边想边吐，前端可以显示
        "思考中…"）。返回值仍是完整的 `LLMResponse`——**流式改变的是
        "你什么时候看到内容"，不是"你拿到什么"**。

        ## ⚠️ 三件关于流式必须知道的事

        **① 缓存命中时不流式。** 缓存是从磁盘读的，本来就一瞬间读完，
        "逐段吐"没有意义。命中时会把整段文本**作为一次 delta** 交出去，
        所以调用方看到的仍然是"有 delta 事件"，只是只有一次。
        （调用方不需要区分这两种情况——见 `api/events.py` 怎么用。）

        **② 流式下 `usage` 依赖 `stream_options.include_usage`。**
        这个客户端会自动带上它。万一服务端不支持，usage 会全是 0，
        成本报表上那一笔就是 0——**`CostBreakdown.is_free` 就是为了让
        "真的是 0"和"没拿到数"区分开。**

        **③ 只在"一个字都没吐出去"时才重试。** 见 `_post_stream` 的约束 ①。
        """
        payload: dict[str, Any] = {
            "model": model or self.model_gen,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        key = _cache_key(payload)
        if self.mode in ("record", "replay"):
            hit = self.cache.get(key)
            if hit is not None:
                # 缓存命中也得"响一下"——见 docstring 第 ① 条。
                # 否则前端会出现"转了半天圈，然后整段答案突然出现"，
                # 那看起来像是页面卡了，而不是"这一轮是从缓存来的"。
                if on_delta is not None and hit.text:
                    on_delta(hit.text)
                if on_reasoning is not None and hit.reasoning_text:
                    on_reasoning(hit.reasoning_text)
                # ⚠️ **不要累加 `hit.usage`。** `usage_total` 的含义是
                # "这个进程实际花了多少 token"，而这一轮是读磁盘、**一分钱没花**。
                # 加上去会让成本报表凭空多出一份钱——而且报表上完全看不出来，
                # 因为那些 token 数看着就是"正常的一次调用"。
                # （这条有测试盯着：`test_cached_response_adds_no_tokens`。）
                return hit
            if self.mode == "replay":
                raise ReplayMiss(
                    f"replay 未命中 cache_key={key}。"
                    "要么先用 --mode record 跑一遍，要么改用 --mode off。"
                )

        streaming = on_delta is not None or on_reasoning is not None
        try:
            resp = (
                self._post_stream(payload, on_delta, on_reasoning)
                if streaming
                else self._post(payload)
            )
        except LLMTruncated:
            # 推理模型的思维链也计入 max_tokens，预算给小了会拿到空正文。
            # 自动翻 4 倍重试一次——**这比把空串抛给调用方好**，
            # 也比让调用方自己猜"要设多大"好。
            #
            # 缓存键仍用**原始** payload 算出的那个：升级是确定性的
            # （同样的输入必然触发同样的升级），所以 replay 时读到的
            # 就是当初升级后的那份响应，复现性不受影响。
            #
            # ⚠️ 升级重试**保持流式**。早期版本这里写死了 `_post`，
            # 结果是"预算不够 → 触发升级 → 用户那边突然断了流、
            # 最后整段答案一次蹦出来"——只在推理模型预算给小的时候才出现，
            # 而且看起来像是"流式坏了"，不像"预算升级了"。
            escalated = {**payload, "max_tokens": min(int(payload.get("max_tokens", 2048)) * 4, 32768)}
            resp = (
                self._post_stream(escalated, on_delta, on_reasoning)
                if streaming
                else self._post(escalated)
            )
        self._record_usage(resp.usage)
        if self.mode == "record":
            self.cache.put(key, resp)
        return resp

    def complete(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        """便利方法：只想要文本。"""
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw).text

    def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        **kw: Any,
    ) -> Any:
        """要求模型回 JSON 并解析。

        用 `response_format={"type": "json_object"}` 让服务端强制合法 JSON——
        比"在 prompt 里求它输出 JSON 再正则抠"可靠得多。解析失败仍然抛错而不是
        返回 None：**静默返回空值会让一批题目悄悄变成垃圾数据**，而下游
        （gold 集）根本看不出来。
        """
        kw.setdefault("response_format", {"type": "json_object"})
        text = self.complete(prompt, system, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"模型返回的不是合法 JSON: {e}\n原始输出前 500 字：\n{text[:500]}") from e

    def complete_many(
        self, requests: list[dict[str, Any]], *, concurrency: int = 6
    ) -> list[LLMResponse]:
        """并发跑一批请求，**按输入顺序返回**。

        顺序必须保——gold 生成时要把结果写回对应的法条，乱序会导致 gold 错位，
        而且错位是**静默**的（题目看起来正常，只是对应错了条）。
        """
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(lambda kw: self.chat(**kw), requests))
