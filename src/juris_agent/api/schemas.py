"""HTTP 的进出格式（F29）。

## 为什么要把「对外格式」和「内部状态」分开

内部有一份 `AgentState`，它字段又多又杂（trace、progress_keys、
invariant_violations……）。**直接把它当响应体吐出去**是最省事的做法，
也是本项目明确不做的事，三个理由：

1. **有些字段是给调试看的，不该出现在契约里。** `progress_keys` 是防死循环用的
   指纹，改了它就会破坏 API；而它随时可能因为"换个哈希算法"而变。
2. **有些字段不能出去。** `on_delta` 是个函数（虽然 `exclude=True` 挡住了），
   `memory` 是会话对象，`tool_messages` 里可能有完整的工具入参。
   靠"记得给每个新字段加 exclude"来防泄漏，迟早会漏一个。
3. **契约要能单独演化。** 前端的字段名和内部字段名不必一样；
   绑死之后，改内部实现就等于破坏接口。

所以这里定义一层显式的出入格式。**代价是多写几十行**，
换来的是"接口不会因为内部重构而变"。

## 一条命名约定

对外的 JSON 字段用**下划线**（`article_key`、`cost_cny`），不用驼峰。
理由：这个项目的 Python 代码、YAML 配置、评测集、报告全是下划线，
前端是唯一会被驼峰诱惑的地方。**只在一个地方换风格，
就是给自己埋一个"取字段时写错了、值是 undefined、页面不报错"的坑。**
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """一条对话消息。形状和 OpenAI / DeepSeek 的 `messages` 一致。"""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """一次提问。

    `session_id` 不传时服务端会新建一段会话并把 id 放在响应里——
    前端把它存下来，下一轮带上，记忆层才能把"第 1 轮说的月薪"带到第 20 轮。
    **不传 session_id 就等于每次都是新会话**，记忆层一行历史都看不到。
    """

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="会话 id。不传则新建一段会话（响应里会给出新 id）。",
    )
    history: list[ChatMessage] | None = Field(
        default=None,
        description=(
            "显式指定历史，**覆盖**服务端会话里的那份。"
            "给两种场景用：① 无状态调用（不想让服务端记任何东西）；"
            "② 复现某次线上问题（把当时的完整历史发过来重跑一遍）。"
        ),
    )
    stream_llm: bool = Field(
        default=True,
        description=(
            "是否让模型**逐字**流式输出。默认开。"
            "关掉之后仍然会按步骤推事件（路由/检索/工具），"
            "只是答案会一次性出现——用它可以对比出'逐字流'值多少体感。"
        ),
    )


class CitationOut(BaseModel):
    """答案引用的一条法条。**前端的"点开看原文"就靠它。**"""

    index: int = 0
    """答案正文里 `[1]` 那个编号。"""
    article_key: str = ""
    citation: str = ""
    """《劳动合同法》第四十七条。"""
    law: str = ""
    article_no: str = ""
    text: str = ""
    score: float = 0.0
    source: str = "hit"
    """`hit` = 检索命中；`neighbor` = 邻条扩展顺带带出来的。"""


class SlotProvenanceOut(BaseModel):
    """一个槽位值，**以及它是第几轮谁说的**。

    这个结构存在的唯一理由是回答用户那句
    「你怎么知道我月薪多少」——答案是一句原话，不是一个推理。
    """

    label: str = ""
    value: str = ""
    turn: int = 0
    quote: str = ""


class MemoryOut(BaseModel):
    """这一轮记忆层实际做了什么。**故意把"丢了什么"也报出来。**"""

    policy: str = "layered"
    tokens: int = 0
    layers: list[dict[str, Any]] = Field(default_factory=list)
    slots: list[SlotProvenanceOut] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    """被预算砍掉的东西 + 原因。答得不全时，先看这里是不是把某层砍光了。"""
    missing: list[str] = Field(default_factory=list)
    """用户还没说、但算不出结果必需的事实（中文标签）。模型会据此**追问**。"""


class TraceOut(BaseModel):
    """一步埋点。**"线上出问题怎么查"的第一手证据。**"""

    step: int = 0
    node: str = ""
    elapsed_ms: float = 0.0
    note: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class CostOut(BaseModel):
    """这一次问答花了多少钱。**估算，不是账单。**

    ⚠️ 这个类**不是** `observability.CostBreakdown`，是一份手工抄的镜像
    （`_to_response` 里用 `CostOut(**cost.model_dump())` 转过来）。
    pydantic 默认**忽略没声明的字段**，所以 `CostBreakdown` 上加了新字段
    而这里忘了加，症状是**接口里那个字段凭空消失、不报错、测试也不红**。
    加字段时两处都要改；`tests/test_api.py` 里有一条测试专门盯着这件事。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    total_cny: float = 0.0
    as_of: str = "unknown"
    basis: str = ""
    """这笔钱怎么算出来的。**必须原样展示**——一个没有出处的金额，
    在被追问时是没有价值的。"""

    billable: bool = True
    """**这一轮的金额是不是真花掉了。**

    为 `False` 时 `total_cny` 仍然可能有值，那是"当初真调会花多少"
    （`replay` 模式、缓存命中），**不是这次的开销**。
    前端和报表都必须先看这个字段再看金额，否则会把重放报成消费。
    详见 `observability/cost.py::CostBreakdown.billable`。"""


class ChatResponse(BaseModel):
    """一次提问的完整回复。**`POST /api/chat` 用它；SSE 的最后一个事件也用它。**"""

    session_id: str
    turn: int = 0
    """这是这段会话的第几轮（从 1 开始）。"""

    answer: str = ""
    citations: list[CitationOut] = Field(default_factory=list)
    """答案引用的法条（已过滤掉模型编造的编号）。"""

    route: str = ""
    route_reason: str = ""

    refusal: dict[str, Any] | None = None
    """拒答时，这里是**三个字段**：类别、原因、建议。见 `prompts.build_refusal`。"""

    finish_reason: str = ""
    finish_detail: str = ""

    citation_report: dict[str, Any] | None = None
    """引用校验的结果：有效几条、编造几条、注入了没引用几条。
    **这是本项目唯一一个零 LLM 成本、可复现的回答质量指标。**"""

    memory: MemoryOut | None = None
    cost: CostOut | None = None
    trace: list[TraceOut] = Field(default_factory=list)
    guard_hits: list[dict[str, Any]] = Field(default_factory=list)
    """撞上防死循环闸门的记录。**空列表才是正常的。**"""

    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    """不影响回答、但你应该知道的（索引与语料不同步、记忆层降级……）。
    和 `errors` 分开是刻意的：混在一起就没法只筛"真的出错了"的那些。"""

    request_id: str = ""


class ToolOut(BaseModel):
    """注册表里的一个工具，**原样摊给界面**。

    这里最有用的一个字段是 `args_schema`——它是 `args_model` 的 JSON Schema。
    界面拿它**现场生成表单**（字段名、类型、说明、枚举选项、上下限全在里面），
    不需要跟后端约好"经济补偿那张表单长什么样"。

    为什么值得这么做：**加一个工具不用改前端。** 在 `builtin.py` 里注册一个
    带 `args_model` 的新工具，界面上就会多出一张表单。反过来（前端手写每张表单）
    的话，新工具上线后界面上根本没有入口，而且**不会报错**——
    它只是"少了一个按钮"，没有任何东西提醒你漏了。

    ⚠️ 唯一的例外是 `refuse_to_answer`：它是给模型用的兜底出口，
    界面上摆一个"我要拒答"的按钮对用户没有意义。但**前端不做白名单**，
    而是照常渲染（它的 `label` 写着"（内部）拒答"）——
    界面上的白名单迟早会和真实的注册表不一致。
    """

    name: str
    label: str = ""
    """人看的短名。空的话前端退回显示 `name`。"""
    group: str = "其他"
    description: str = ""
    """**写给模型的那段**（含"不适用场景"）。前端折叠显示，不删——
    它是"这个工具什么时候不该用"的唯一书面说明。"""
    args_schema: dict[str, Any] = Field(default_factory=dict)
    idempotent: bool = True
    timeout_seconds: float | None = None


class ToolRunOut(BaseModel):
    """按一次界面上那个"运行"按钮的结果。

    ## 字段和 `tool_result` 事件**刻意保持一致**

    同一件事（调用一个工具）在两条路上发生：Agent 自己调（走 SSE 的
    `tool_result` 事件）、用户手动点（走这个接口）。两边的字段名如果不一样，
    前端就得写两套渲染，而两套里必然有一套先烂——那时
    "界面上看到的"和"Agent 真跑出来的"就不是一回事了，
    而这个项目一直在强调的正是"看到的就是真跑的"。

    ## ⚠️ 入参写错**不返回 4xx**

    `status="failed"` + `error` 里的结构化反馈才是正确形状。理由：
    入参校验失败是**这个工具的常规业务结果**（Agent 那边每天都遇到，
    它靠这份反馈换策略），不是"HTTP 请求有问题"。
    返回 400 的话，前端得把"请求错了"和"参数不合法"分成两条路处理，
    而后者其实和"工具执行失败"是同一条路——**多分的这一条就是将来会忘的那条。**
    """

    tool: str
    status: str = ""
    """`ok` / `degraded` / `failed`。`degraded` = 走了降级路径，结果可用但精度下降。"""
    attempts: int = 0
    elapsed_ms: float = 0.0
    idempotency_hit: bool = False
    breaker_state: str = "closed"
    payload: Any = None
    """结果的 JSON 形式（`ToolOutcome.payload()`）。**失败时为 None。**"""
    error: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class HealthOut(BaseModel):
    """服务自检。**没有它，前端报错时你分不清是服务没起还是请求写错了。**

    ⚠️ **它比 `Runtime.health()` 松，这是故意的。** `health()` 返回的是一个
    手写 dict，字段会随开发增删；如果用 `response_model=HealthOut` 严格校验，
    每加一个 health 字段都得同步改这里，**忘了改就会被静默丢掉**——
    而"健康检查少报一个字段"是最难察觉的一类问题（因为接口照样返回 200）。
    所以 `/api/health` **不挂 response_model**，把 dict 原样发出去；
    这个类的用处是**给前端当类型说明**，不是当校验闸门。
    多出来的字段（`uptime_seconds` / `llm_mode` / `retrieval` / `memory` /
    `sessions` 等）以 `Runtime.health()` 为准。
    """

    model_config = ConfigDict(extra="allow")

    ok: bool = True
    ready: bool = False
    """语料、索引、模型**都**装好了吗。没装好照样能起服务，但问什么都答不了。"""
    problems: list[str] = Field(default_factory=list)
    """装配时真实遇到的问题。**空列表才代表一切正常**——非空时要显示出来。"""


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "CitationOut",
    "CostOut",
    "HealthOut",
    "MemoryOut",
    "SlotProvenanceOut",
    "ToolOut",
    "ToolRunOut",
    "TraceOut",
]
