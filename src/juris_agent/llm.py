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
                    raise LLMAuthError(f"鉴权失败 HTTP {r.status_code}: {r.text[:200]}", status=r.status_code)
                if r.status_code == 429:
                    last = LLMRateLimit(f"限流 HTTP 429: {r.text[:200]}", status=429)
                elif 400 <= r.status_code < 500:
                    # 4xx 不重试。抛出去让调用方看见真实的错误原因。
                    raise LLMBadRequest(f"请求错误 HTTP {r.status_code}: {r.text[:300]}", status=r.status_code)
                else:
                    last = LLMServerError(f"服务端错误 HTTP {r.status_code}: {r.text[:200]}", status=r.status_code)

            if attempt < self.max_retries and last is not None and last.retryable:
                time.sleep(min(2.0 ** attempt, 8.0))
                continue
            break
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
                return hit
            if self.mode == "replay":
                raise ReplayMiss(
                    f"replay 未命中 cache_key={key}。"
                    "要么先用 --mode record 跑一遍，要么改用 --mode off。"
                )

        try:
            resp = self._post(payload)
        except LLMTruncated:
            # 推理模型的思维链也计入 max_tokens，预算给小了会拿到空正文。
            # 自动翻 4 倍重试一次——**这比把空串抛给调用方好**，
            # 也比让调用方自己猜"要设多大"好。
            #
            # 缓存键仍用**原始** payload 算出的那个：升级是确定性的
            # （同样的输入必然触发同样的升级），所以 replay 时读到的
            # 就是当初升级后的那份响应，复现性不受影响。
            escalated = {**payload, "max_tokens": min(int(payload.get("max_tokens", 2048)) * 4, 32768)}
            resp = self._post(escalated)
        with self._usage_lock:
            self.usage_total = self.usage_total + resp.usage
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
