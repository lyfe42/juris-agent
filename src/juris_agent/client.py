"""对外的调用门面：**让"用一下这个 Agent"从五行变成一行。**

## 这个文件为什么存在

在这之前，想跑一个问题要自己攒出这样一串东西：

```python
from juris_agent.agent.graph import make_runner
from juris_agent.config import get_settings
from juris_agent.llm import LLMClient
from juris_agent.retrieval.build import build_default_retriever

settings = get_settings()
retriever = build_default_retriever(settings=settings)   # 加载约 1GB 的嵌入模型
llm = LLMClient()                                        # 没有 key 这里会抛
runner = make_runner(retriever=retriever, llm=llm, settings=settings)
llm.begin_turn()                                         # 不开账本，成本就记不到
state = runner.ask("被辞退有补偿吗？")
```

**这五行里藏着三个只有读过源码的人才知道的约束**：

1. `build_default_retriever` 是重活（加载模型 + 可能建索引，几秒到几分钟），
   必须**进程级只做一次**，不能每个问题做一次；
2. `LLMClient` 没有 key 时会抛 `LLMAuthError`，但那句话埋在第三个 import 里；
3. `begin_turn()` 必须在**跑那一轮的那个线程里**调，否则成本读到的是 0
   （而且是静默的 0，见 `llm.LLMClient.begin_turn` 的 docstring）。

一个只想试试的人，没有理由知道这三件事。**门面就是把它们收进来。**

## 为什么不是"再写一套装配"，而是**复用 `build_runtime()`**

`api/runtime.py` 里已经有一套"把重东西造一次"的装配，`api/app.py` 的那 43 个
接口契约测试全压在它上面。这里如果再写一套并行的装配，会出现两个必然的后果：

- **两套会漂移**：哪天检索器多了一个参数、或者 LLM 客户端换了构造方式，
  改了一处忘了另一处，症状是"Web 能用、库里调用报错"，而两边的代码看着都对。
- **门面这条路没有测试覆盖**：`build_runtime` 的每一条降级分支都被 API 测试
  盯着（缺 key、缺模型、缺语料各有断言），新写一份就是从零开始漏。

所以这里只做三件事：**懒加载、把 `problems` 变成异常、把 `AgentState`
变成好读的 `Answer`**。装配本身一律交给 `build_runtime()`。

## 用法

```python
from juris_agent import ask

a = ask("公司裁员，我干了 3 年月薪 1 万，能赔多少？")
print(a.text)        # 答案 + 引用法条（带原文）
print(a.cost_text)   # ¥0.0123（这一轮真的花了）
```

多轮（同一个 `session_id` 就是同一段对话）：

```python
from juris_agent import JurisAgent

agent = JurisAgent()
agent.ask("我在西安，月薪 12000，被裁了", session_id="s-1")
agent.ask("那到底能赔多少？", session_id="s-1")   # 记得住月薪和地点
agent.close()        # 关掉 httpx 连接池
```

## ⚠️ 这个门面**不改变**任何一项行为

它不加兜底、不重试、不改写答案、不吞异常。答案里的 `route` / `finish_reason`
/ `citations` / `refusal` 全部原样来自图里，**一个字段都没加工**——
项目里所有关于"准确率、拒答率、引用校验"的报表跑的都是同一条路，
门面要是顺手做了点什么，那些数字就都成了"报表测的是一回事、
用户用的是另一回事"。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from juris_agent.api.runtime import Runtime, build_runtime
from juris_agent.config import Settings
from juris_agent.observability import describe_cost, estimate_cost, get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #


class NotReady(RuntimeError):
    """装不起来，或者缺东西，反正现在答不了问题。

    ## 为什么这里是**抛异常**，而服务那边只是"记进 `problems` 然后带病启动"

    这不是两套标准，是**两种场景需要相反的行为**：

    - 服务（`api/`）**做不出"拒绝启动"这个选择**。它必须能起来，
      因为演示的人可能只想看看前端长什么样、引用能不能点开，
      而那时模型还没装完（详见 `api/runtime.py` 的模块 docstring）。
      所以它降级、把问题写进 `/api/health`，让人**看得见**。
    - 库里调用的人**没有 health 页面可看**。拿到一个静默降级的东西，
      症状是"`ask()` 返回了空答案"或者"报了个看不懂的 AttributeError"，
      而真正的原因（少一个环境变量）在**另外一个地方**。
      这种排查最耗时间的地方在于它把矛头指向了错误的嫌疑人。

    所以：**库这边把 `problems` 原样贴在异常消息里**，
    缺什么、缺在哪一步，第一眼就能看到。想要"带病也让我看看"的，
    用 `agent.health()` —— 它**永远不抛**。
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems) or "  - （没有记录到原因）"
        super().__init__(
            "juris-agent 没能装配起来，现在答不了问题。装配过程中遇到的问题：\n"
            f"{body}\n"
            "排查顺序见 README 的「装不起来怎么办」，或者跑 `juris doctor` 看完整清单。\n"
            "⚠️ 修完配置后要新建一个 JurisAgent（或调 .close()）——"
            "失败的装配结果会被记住，理由见 JurisAgent._ensure 的 docstring。"
        )


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Citation:
    """答案引用的一条法条。**`index` 就是答案正文里 `[1]` 那个编号。**

    `article_key` 是稳定的字符串 id（如 `labor_contract_law#47`），
    拿它可以回到 `data/laws.jsonl` 里找那条原文——
    这就是"引用可点开"的全部机制，不需要任何额外索引。
    """

    index: int
    key: str
    """`article_key`。⚠️ 字段名不叫 `article_key` 是为了打字短：
    `for c in a.citations: print(c.key)` 读起来比 `c.article_key` 顺。"""
    text: str
    """《劳动合同法》第四十七条 —— 给人看的那一行。"""
    law: str = ""
    article_no: str = ""
    body: str = ""
    """法条原文（截断过的）。**完整原文在 `Answer.state` 里**，见 `_citation_of`。"""
    score: float = 0.0
    source: str = "hit"
    """`hit` = 检索命中；`neighbor` = 邻条扩展顺带带出来的（前后各一条）。"""

    def __str__(self) -> str:
        return self.text


@dataclass
class Answer:
    """一轮问答的结果。

    ## 为什么是 dataclass 而不是直接回 `AgentState`

    因为 `AgentState` 有 35 个字段，其中大半是给图和测试看的中间量
    （`on_delta`、`seen_state`、`memory_layers`…）。调用方想知道的是
    「答了什么、引了哪几条、为什么拒答、花了多少钱」这四件事。

    ⚠️ **但原始状态一个字都没丢**，在 `state` 里。要看 trace、guard_hits、
    引用校验明细的，直接读它——**门面是视图，不是副本**。
    """

    question: str = ""
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)

    route: str = ""
    """`law` / `calc` / `out_of_scope` / `chitchat`。"""
    route_reason: str = ""
    finish_reason: str = ""
    """`answered` / `refused` / `guard_*` / `error`。"""
    finish_detail: str = ""
    refusal: dict[str, Any] | None = None
    """拒答时是给用户看的那段话（也是答案正文）。超纲/闲聊时由路由决定。"""

    cost: Any = None
    """`observability.CostBreakdown`。**不要直接读 `total_cny`**，
    先看 `billable`——三种"0 元"含义完全不同，见 `describe_cost`。"""
    steps: int = 0
    tokens: int = 0
    warnings: list[str] = field(default_factory=list)
    """这一轮**成功返回了答案、但有一件事没成**（目前只有一种：
    多轮会话写回被 CAS 拒了）。见 `_commit_turn`。"""

    state: Any = None
    """原始 `AgentState`。要 trace、引用校验明细、记忆来源的看这里。"""

    # -- 好读的派生属性 ---------------------------------------------------- #

    @property
    def refused(self) -> bool:
        """是不是拒绝了（而不是超纲/闲聊）。

        ⚠️ **不看 `refusal` 字段是不是 None**：因为它**会漏键**——
        实测 `AgentState` 声明了 35 个字段而 `values` 给出的 dict 里少了它
        （见 `api/app.py` 的 `_g` docstring）。这里统一以 `finish_reason` 为准，
        它是每轮必有、且只有 `refused` 这一个值表示"拒答"。
        """
        return self.finish_reason == "refused"

    @property
    def ok(self) -> bool:
        """这一轮干干净净地答完了：没拒答、没被闸门截、没出错。"""
        return self.finish_reason == "answered"

    @property
    def cost_text(self) -> str:
        """一句话说清这一轮花了多少。**用 `describe_cost`，不自己拼字符串。**"""
        return describe_cost(self.cost)

    @property
    def text(self) -> str:
        """给人看的整段：答案 + 引用法条（带一小段原文）。

        ⚠️ **这是"演示给人看"的格式，不是"机器读"的格式。**
        下游要处理的话读 `answer` / `citations` / `state`，
        别去 parse `text`——法条原文是截断过的，截断位置会随字数变化。
        """
        lines = [self.answer.strip()]
        if self.citations:
            lines.append("")
            lines.append("引用法条：")
            for c in self.citations:
                head = f"[{c.index}] {c.text}"
                body = _oneline(c.body, _TEXT_BODY_CHARS)
                lines.append(f"{head} —— {body}" if body else head)
        elif self.route in ("out_of_scope", "chitchat"):
            # 这两种路由**本来就没有引用**（超纲直接拒、闲聊直接回），
            # 打一行说明比留白好——不然看着像"检索失败了"。
            lines.append("")
            lines.append("（这类问题不查法条，所以没有引用。）")
        return "\n".join(lines)

    def __str__(self) -> str:
        """`print(a)` 和 `print(a.text)` 等价。**省掉一次"."**"""
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """可 JSON 序列化的一份。**给"把结果存下来做报表"用。**

        ⚠️ 这里面**没有** `state`——它含 `on_delta` 之类的回调对象，
        不一定能序列化，而且体积是答案的几十倍。要 trace 的话单独取。
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "route_reason": self.route_reason,
            "finish_reason": self.finish_reason,
            "finish_detail": self.finish_detail,
            "refused": self.refused,
            "steps": self.steps,
            "tokens": self.tokens,
            "warnings": list(self.warnings),
            "cost": self.cost.model_dump() if self.cost is not None else None,
            "citations": [
                {
                    "index": c.index,
                    "key": c.key,
                    "citation": c.text,
                    "law": c.law,
                    "article_no": c.article_no,
                    "score": c.score,
                    "source": c.source,
                }
                for c in self.citations
            ],
        }


# `text` 里法条原文截到多少字。
#
# 为什么截：`inject_k` 默认 8 条，每条约 200~400 字，全打出来一屏装不下，
# 而演示时要"一眼看到引用的是什么"。
# 为什么不截得更短：太短会把"经济补偿按劳动者在本单位工作的年限"这种
# 关键前提切掉，看着像引错了法条。
_TEXT_BODY_CHARS = 60


def _oneline(s: str, limit: int) -> str:
    """压成一行并截断。**换行会毁掉"一条一行"的对齐**，所以换行也当分隔符。"""
    flat = " ".join((s or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


# --------------------------------------------------------------------------- #
# 门面本体
# --------------------------------------------------------------------------- #


class JurisAgent:
    """劳动法咨询 Agent 的门面。**进程里造一个，之后反复用。**

    ## 为什么是"造一个、反复用"而不是"每个问题造一个"

    因为构造这一侧有一条**不能重复付**的成本：检索器要加载约 1GB 的嵌入模型
    （还要建/读一次索引），这是几秒到几分钟。每问一个问题加载一次的话，
    单次问答的响应时间会被这一件事主导，而它和问题本身毫无关系。

    同理还有 `ToolExecutor` 里的熔断器和幂等缓存——它们**要靠历史才学得到东西**。
    每轮新建一个等于每次都从"什么失败都没见过"开始，熔断器就永远不会打开。

    ## 懒加载：`JurisAgent()` 本身**很便宜**

    真正的装配推迟到第一次 `ask()` / `chat()`。这样：

    - 只想 `from juris_agent import JurisAgent` 看一下有什么方法，不用等模型加载；
    - 单测里注入假检索器（`JurisAgent(retriever=..., llm=...)`）**完全不碰模型**；
    - 装配失败发生在**真正要用的那一刻**，而不是在一个看起来无关的 import 上。

    ## 线程安全

    装配那一步加了锁（`_lock`）。**不是为了并发提问**，是为了防止两个线程
    同时走到第一次装配：两个线程各加载一份 1GB 的模型，几秒内显存翻倍，
    然后其中一份立刻被丢掉——白花的时间和内存都是几秒级的，
    而且**不报错**，只是"第一次变慢了"。

    `ask()` 本身是同步的、可重入的；`LLMClient` 的账本是**线程本地**的，
    所以两个线程各问各的不会串账。但**同一个 `session_id` 被两个线程同时用**
    会撞上 CAS（后写的那个拿到 `SessionConflict`）——这是设计如此，
    不是 bug，见 `api/sessions.py` 的模块 docstring。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        runtime: Runtime | None = None,
        retriever: Any = None,
        llm: Any = None,
        new_memory: Callable[[], Any] | None = None,
    ) -> None:
        """**所有参数都可注入——这不是为了灵活，是为了能测。**

        和 `build_runtime` 里的理由一字不差：一个只有"从零装配真实依赖"
        这一条路的门面，测试里就只剩 ① 真的加载模型 + 真的调 API
        （慢、要钱、不可复现）或者 ② 不测。

        - `runtime=`：直接给一颗装好的依赖树，**连懒加载都跳过**。
        - `retriever=` / `llm=`：塞假的进去，其余照常走 `build_runtime`。
        - `settings=`：换配置（比如临时把 `llm_mode` 设成 `replay` 省钱）。
        """
        self._settings = settings
        self._runtime = runtime
        self._retriever = retriever
        self._llm = llm
        self._new_memory = new_memory
        self._lock = threading.Lock()

    # -- 生命周期 ---------------------------------------------------------- #

    def _ensure(self) -> Runtime:
        """拿到装好的 runtime，没装就装。**失败过的装配结果会被记住。**

        ## ⚠️ 为什么"失败也要记下来"（而不是失败后把 `_runtime` 清空好重试）

        这是想清楚了才这么写的。清空看起来更友好（"你补上 key 再问一次就好了"），
        但它会把**一次清楚的错误**变成**每次都慢几秒的错误**：
        装配失败最常见的原因是模型没装/没下，而这条路**每试一次要花几十秒**
        才开始报错。于是 `for q in questions: ask(q)` 这种循环会变成
        "每个问题先卡 30 秒再失败"，而真正的原因（模型路径不对）
        被埋在第 N 次重复里。

        记住之后：**第一次失败给出完整原因，后面每次立刻失败**，
        并且异常消息里直接写了恢复动作（`close()` 或新建一个）。
        这个取舍和 `build_runtime` 那边是一致的——
        那边是"失败也起服务"，这边是"失败也记住失败"，都是**别让同一个原因重复付代价**。
        """
        # 第一次检查**故意在锁外面**：装好之后（99.9% 的调用）就不该再抢锁。
        # 这是个双检锁——不加外面这一次的话，每问一个问题都要抢一次全局锁，
        # 而多轮并发下这就成了一个纯粹的串行点。
        if self._runtime is not None:
            return self._check(self._runtime)

        with self._lock:
            if self._runtime is None:
                self._runtime = build_runtime(
                    self._settings,
                    retriever=self._retriever,
                    llm=self._llm,
                    new_memory=self._new_memory,
                )
            rt = self._runtime
        return self._check(rt)

    @staticmethod
    def _check(rt: Runtime) -> Runtime:
        if not rt.ready:
            raise NotReady(rt.problems)
        return rt

    @property
    def runtime(self) -> Runtime:
        """装好的依赖树。**第一次访问会触发装配**（可能几秒到几分钟）。"""
        return self._ensure()

    @property
    def ready(self) -> bool:
        """**不触发装配**，也不抛异常。

        和 `health()` 的区别：这个只回答"现在是不是已经装好了"，
        给"要不要现在就问"的判断用；`health()` 会去装一次，
        拿完整的诊断信息。**两个都需要**：
        判断用的那个不能有副作用（否则"检查一下"就等于"加载模型"），
        诊断用的那个必须去装（否则它答不了"为什么装不起来"）。
        """
        return self._runtime is not None and self._runtime.ready

    def health(self) -> dict[str, Any]:
        """诊断信息。**永远不抛异常**——这正是它的用途。

        装上就用 `Runtime.health()`；装不上就把 `problems` 包成同样形状返回，
        外加 `error` 那一栏说明异常消息。**两种情况的返回形状一样**，
        所以 `juris doctor` 和调用方都不用写分支。
        """
        try:
            rt = self._ensure()
        except NotReady as e:
            return {
                "ok": True,
                "ready": False,
                "problems": e.problems,
                "error": str(e),
            }
        data = rt.health()
        # ⚠️ 补一个**服务端没有、库这边才有**的字段：
        # 服务在 `/api/health` 里报 sessions 的统计就够了，
        # 而库里调用的人想知道的是"这次装的是哪个模型"——
        # 因为 `settings=` 是可以传的，同一台机器上跑两个不同配置的实例很常见。
        data["settings"] = {
            "llm_mode": rt.settings.llm_mode,
            "env": rt.settings.env,
            "model_gen": rt.settings.model_gen,
            "model_judge": rt.settings.model_judge,
            "embedding_model": rt.settings.retrieval.embedding_model,
        }
        return data

    def close(self) -> None:
        """关掉占着系统资源的东西，并丢掉这份 runtime。**用完调一次，或者用 `with`。**

        ## 只关 LLM 客户端

        它持有一个 httpx 连接池，是真需要显式释放的东西。
        检索器没什么可关的——它就是几个 numpy 数组加一份模型权重，
        进程退出时解释器自己回收（理由见 `api/runtime.py` 的 `Runtime.close`）。

        ## 关完之后**还能继续 `ask()`**

        因为引用被丢掉了，下次用会重新装配一遍。这个行为是刻意的：
        调用方在 REPL 里随手 `close()` 一下，不应该让这个对象**永久报废**——
        那会逼着每个人去记住"什么时候还能用、什么时候不能用"。

        ⚠️ **但代价要说清楚**：重装那一次会**重新加载嵌入模型**（几秒到几分钟）。
        所以"每问一句就 close 一下"是错的，close 该在**收尾**时调。

        ⚠️ **`runtime=` 注入进来的那份回不来了。** 一个已经构造好的 `Runtime`
        对象没法"照着再构造一次"，所以重装的路径只能是
        `build_runtime(settings, ...)`（也就是**真**依赖）。
        **测试里写了 `close()` 之后还想继续用假件的话，得再注入一次**。

        ## 为什么**没有**一个单独的 `reset()`

        一开始两个都写了，但它们的实现逐行相同。**两个名字一个动作**是个坑：
        读代码的人会去找它们的区别，找不到，然后开始怀疑自己看漏了什么；
        更糟的是，其中总有一个会在某次改动之后被漏掉，
        于是"reset 之后行为不一样"这种 bug 就长出来了。
        想表达"我改了配置要重来"，那也还是 `close()`——
        把它读成"把这一份丢掉"就对了。
        """
        with self._lock:
            rt, self._runtime = self._runtime, None
        if rt is not None:
            rt.close()

    def __enter__(self) -> JurisAgent:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- 提问 -------------------------------------------------------------- #

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Answer:
        """问一个问题，等它答完。

        ## 两种模式，看 `session_id` 给不给

        - **不给（默认）**：一问一答，无状态。多轮上下文的"历史"由
          `history=` 这个参数自己带（`[{"role": "user", "content": ...}, ...]`）。
        - **给一个 id**：把这个 id 当成一段对话的钥匙。同一个 id 的第二次
          `ask` 会**自动带上前面几轮的原文**，并且复用同一个记忆管理器
          （槽位表：月薪、工龄、地点）。这就是 CLI 的 `juris chat` 走的路。

        ⚠️ **不要自己拿一个 dict 到各处共享当"历史"。**
        记忆管理器**绝不能跨会话共用**——共用了就会出现"A 用户的月薪
        出现在 B 用户的答案里"（见 `api/runtime.py` 那张表）。
        用 `session_id` 的话这件事由 `SessionStore` 管，不用你操心。

        ## `on_delta` / `on_reasoning`：边生成边看

        传了就变成流式——答案一字一字回调给你，而不是等全部生成完。
        **一轮只影响这一轮**：它们进的是这一轮的状态，不进共享依赖，
        所以并发提问不会串（理由见 `agent/state.py` 上那两个字段）。

        ⚠️ **回调是在同一个线程里同步调的**，在里面做重活会拖慢生成。
        要更新终端的话，攒几个字符再刷一次（`api/app.py` 里就是这么做的）。
        """
        rt = self._ensure()
        runner = rt.runner

        # --- 会话层面的准备 ------------------------------------------------- #
        session = None
        expected_version = 0
        if session_id is not None:
            session = rt.sessions.get_or_create(session_id)
            # ⚠️ **版本号必须现在读、写回时用**。等写回时再从 session 上读，
            # 读到的永远是"当前的"，校验就恒等于通过——等于没有校验。
            # 理由见 `SessionStore.commit` 的 docstring，那里把它写成
            # 必传参数就是为了让人没法绕过。
            expected_version = session.version
            hist = history if history is not None else session.messages()
            memory = session.memory
        else:
            hist = list(history or [])
            memory = None

        # --- 跑一轮 --------------------------------------------------------- #
        llm = rt.llm
        # ⚠️ **开账本必须在"待会儿跑图"的那个线程里**，也就是这里。
        # 账本是**线程本地**的：在别的线程开、在这里读，读到的是全 0，
        # 而且它**不报错**（`turn_usage` 对没开过账的线程设计成返回 0）。
        # 后果是成本报表上金额永远走"按 token 粗估"那条退路——
        # 那个数看着完全合理，所以这个 bug 在真机上跑了一版都没被发现。
        if llm is not None and hasattr(llm, "begin_turn"):
            llm.begin_turn()

        state = runner.ask(
            question,
            history=hist,
            memory=memory,
            on_delta=on_delta,
            on_reasoning=on_reasoning,
        )

        # ⚠️ **账要立刻抄下来**，不能等函数快结束时再读。
        # 目前这里是同步一条线，抄不抄都一样；但 `begin_turn` 的语义是
        # "开一本新账"，只要将来在中间插一个别的 LLM 调用（比如多轮摘要），
        # 账本就被重置了，而成本会**安静地变成那个小数字**。
        # 先抄下来，等于把"读到错账"这件事从"哪天会踩"变成"不可能踩"。
        usage = getattr(llm, "turn_usage", None) if llm is not None else None

        cost = _cost_of(rt.settings.model_gen, state, usage)

        # --- 写回会话 ------------------------------------------------------- #
        warnings: list[str] = []
        if session is not None:
            warnings = _commit_turn(rt, session, expected_version, question, _attr(state, "answer", ""))

        return _to_answer(question, state, cost, warnings)


def _attr(state: Any, key: str, default: Any) -> Any:
    """从状态里读一个字段，`dict` 和对象两种形状都吃，且把 `None` 当缺省。

    ## 为什么不直接用 `getattr`（这是实测出来的，不是防御性编程）

    `AgentRunner.ask()` 返回的**实测是 `AgentState` 实例**，所以在这个门面里
    其实走不到 dict 那一支。保留它有两个理由：

    1. `runner.ask` 的返回类型在 docstring 里写的是"归一成 `AgentState`"，
       也就是**它做过一次判断**（LangGraph 有 checkpointer 时返回 dict）。
       这个分支存在这件事本身说明"形状"不是我们能单方面决定的。
    2. 更关键的是 `default if v is None else v` 这一句。
       `AgentState` 的字段是 **pydantic 模型**，`refusal` / `citation_report`
       这些字段的默认值就是 `None`；而 `getattr(state, "answer", "")`
       在字段存在但值为 `None` 时会返回 `None`，不是 `""`。
       于是 `answer` 变成 `None`，`Answer.text` 里那句 `self.answer.strip()`
       会抛 `AttributeError`——**报错信息指向 `text`，
       而真正的原因在三个函数之外**。

    所以这里统一走一个"容忍两种形状、且对缺键和 None 返回默认值"的读法。
    """
    v = state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)
    return default if v is None else v


def _cost_of(model: str, state: Any, usage: Any = None) -> Any:
    """这一轮的 token 与金额。

    ## 为什么在门面里**再写一遍**，而不是去调 `api/app.py` 的 `_cost_of`

    对称的写法是把它从 `api/app.py` 提到公共模块、两边都 import。
    没这么做，理由是这个函数在两边**读的东西不同**：

    - `api/app.py` 那版收的是 `Runtime`，因为服务那边**必然**有 runtime；
    - 这里收的是 `model` 一个字符串，因为库里调用的人可能压根没有 runtime
      （注入假 LLM 做测试的情形）。

    把它硬拧成一个函数，就得让参数变成"runtime 或者 model 二选一"，
    而那种签名每个人都要读两遍才敢传。**代价是这段逻辑有两份。**

    ⚠️ **所以下面这两条规则必须两边同时改**（它们的来源都是踩过的坑，
    详细理由见 `api/app.py` 的 `_cost_of`）：

    1. `usage` 有数就用 `estimate_cost` 按分项算；
    2. 走退路时**必须显式把 `billable` 置 False**。

    第 2 条是重点。`replay` 模式命中缓存时 `LLMClient` **刻意不记账**，
    于是 `turn_usage` 是一本空账，代码掉到退路，用状态里 `tokens_used`
    估出金额——但那个 token 数是**当初真调那一次**留下的。
    不置 `billable=False` 的话，会写出"重放一遍花了 0.13 元"这种
    自相矛盾的话，而且 **`¥0.138040` 看着太正常了，肉眼看不出来**。
    """
    if usage is not None and (usage.total_tokens or usage.prompt_tokens):
        return estimate_cost(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_hit_tokens=usage.prompt_cache_hit_tokens,
            cache_miss_tokens=usage.prompt_cache_miss_tokens,
        )

    # 退路：这一轮没有计过费的调用（拒答、闲聊、或者缓存重放）。
    total = int(_attr(state, "tokens_used", 0) or 0)
    c = estimate_cost(model=model, prompt_tokens=0, completion_tokens=total)
    c.billable = False
    c.basis = (
        "按状态里记录的 token 总数估算（未取到分项）" if total else "这一轮没有调用模型"
    )
    return c


def _commit_turn(
    rt: Runtime,
    session: Any,
    expected_version: int,
    question: str,
    answer: str,
) -> list[str]:
    """把这一轮写回会话。**写不回去也照常把答案给人，但必须说出来。**

    ## 为什么不重试、也不吞掉

    和 `api/app.py` 里的处理一字不差，因为这里的取舍是同一件事：

    - **不重试**：重试要先重新读一遍历史、再跑一整轮图——那是第二次付钱。
      而且它掩盖了一个事实：**有人正在并发改这段会话**，重试只是让
      "你俩到底谁说的话留下来了"变得更难预测。
    - **不吞掉**：吞掉就等于回到"后写的盖掉先写的"那个 bug，
      而且是静默的。用户下一轮会少看到这一轮的历史，
      不知道的话会以为"模型记性不好"。

    所以答案照常给（钱都花了），把冲突写进 `Answer.warnings`。

    ## ⚠️ 这里**没有加锁**，是刻意的

    `Session.lock` 是 `asyncio.Lock`，在同步代码里用不了。
    而"再加一把 `threading.Lock`"会给人一种错觉：好像这把锁能防住
    多进程、防住"某条新路径忘了加锁"。它一样防不住——
    而**版本号（CAS）两样都防得住**，因为它校验的是"这期间有没有人改过"，
    与改的人在哪、用没用锁无关。

    已有的机制够用，就不要再造第二个。这条正是 `api/sessions.py`
    模块 docstring 里"为什么锁和版本号要同时有"的推论：
    **加了锁也仍然需要版本号**，那么在没有锁的场景里，版本号就是全部。
    """
    from juris_agent.api.sessions import SessionConflict

    try:
        rt.sessions.commit(
            session, expected_version=expected_version, question=question, answer=answer
        )
    except SessionConflict as e:
        log.warning("会话写回被拒", session_id=session.session_id)
        return [str(e)]
    return []


def _citation_of(injected: Any, body_limit: int = 0) -> Citation:
    """`InjectedArticle` → `Citation`。

    ⚠️ **编排序 —— `valid` 列表里的编号直接用来查 `injected` 字典**，
    这里**不做任何重新编号、也不做去重排序**。
    真实答案里的编号是模型照着 prompt 里 `[i]` 抄下来的，
    重排会让"答案里的 [3]"和"引用列表里的第 3 条"变成两个东西——
    那正是这个项目"引用可校验"要防的事。
    """
    return Citation(
        index=int(injected.index),
        key=str(injected.article_key),
        text=str(injected.citation),
        law=str(injected.law),
        article_no=str(injected.article_no),
        body=_clip(str(injected.text), body_limit),
        score=float(injected.score or 0.0),
        source=str(injected.source or "hit"),
    )


def _clip(s: str, limit: int) -> str:
    """截断到 `limit` 个字符。`limit<=0` 表示不截。

    ⚠️ **不在这里压换行。** 压换行是"排版"这件事的决定，
    由 `Answer.text` 那一层做（它知道自己要一行一条）。
    在这里顺手压掉的话，`Answer.state` 里那条法条的原文
    就和语料里的**不再逐字相同**——而"逐字相同"是这个项目
    所有引用校验（`citation_report`）的前提。
    """
    if limit and limit > 0 and len(s) > limit:
        return s[:limit] + "…"
    return s


def _to_answer(question: str, state: Any, cost: Any, warnings: list[str]) -> Answer:
    """`AgentState` → `Answer`。**这是唯一一处做这个转换。**

    ## 引用是从哪来的（两条路，必须和 `api/app.py` 保持一致）

    - **有 `citation_report`**（正常答完的那一轮必有）：用它的 `valid`
      列表——那是**程序算出来的**"答案里出现的编号，且确实落在注入范围内"。
      零 LLM、零成本、可复现，见 `agent/state.py` 的 `CitationReport`。
    - **没有报告**（拒答、闲聊、超纲）：退回按 `citations` 里的 key 去匹配。
      这些路由不注入法条，所以正常情况下列表是空的；
      留着这支是为了"拒答但引用了"这种异常情形不至于**整段引用凭空消失**
      （凭空消失比少几条更糟：它会让人以为检索坏了）。

    ⚠️ **不取 `citation_report` 里的 `invalid`。** 那些是模型编出来的编号
    （引用了一条不存在的法条）。把它们放进引用列表，等于**替模型圆谎**。
    它们仍然在 `state.citation_report.invalid` 里，要统计"幻觉率"就取那一栏。
    """
    injected_raw = _attr(state, "injected", []) or []
    injected = {int(_attr(a, "index", 0)): a for a in injected_raw}
    report = _attr(state, "citation_report", None)

    if report is not None:
        valid = list(_attr(report, "valid", []) or [])
        picked = [injected[int(i)] for i in valid if int(i) in injected]
    else:
        keys = set(_attr(state, "citations", []) or [])
        picked = [a for a in injected.values() if _attr(a, "article_key", "") in keys]

    return Answer(
        question=question,
        answer=str(_attr(state, "answer", "") or ""),
        # ⚠️ `body_limit=0`：**引用里的原文不截断**，`Answer.text` 排版时才截。
        # 截在这里的话，谁都没法从 `Answer` 上拿到完整的法条——
        # 而"点开看原文"是这个产品的主打功能之一。
        citations=[_citation_of(a) for a in picked],
        route=str(_attr(state, "route", "") or ""),
        route_reason=str(_attr(state, "route_reason", "") or ""),
        finish_reason=str(_attr(state, "finish_reason", "") or ""),
        finish_detail=str(_attr(state, "finish_detail", "") or ""),
        refusal=_attr(state, "refusal", None),
        cost=cost,
        steps=int(_attr(state, "steps", 0) or 0),
        tokens=int(_attr(state, "tokens_used", 0) or 0),
        warnings=list(warnings),
        state=state,
    )


# --------------------------------------------------------------------------- #
# 模块级便利函数：单例
# --------------------------------------------------------------------------- #

_default: JurisAgent | None = None
_default_lock = threading.Lock()


def get_agent(settings: Settings | None = None) -> JurisAgent:
    """拿到进程级的默认 agent，没有就造一个。

    ⚠️ **`settings` 只在"造的那一次"生效。** 已经造好之后再传一个不同的
    settings 进来，**不会**改变任何东西——这是刻意的：
    如果它悄悄重建，那"加载了几 GB 的模型"就会在某个看起来无害的调用点
    悄悄发生一次。要换配置就显式调 `close()`（它会丢掉当前这份），把代价摆在明面上。
    """
    global _default
    if _default is not None:
        return _default
    with _default_lock:
        if _default is None:
            _default = JurisAgent(settings)
    return _default


def ask(question: str, **kwargs: Any) -> Answer:
    """`get_agent().ask(question, **kwargs)`。**给脚本和交互式试用的一行入口。**

    等价于：

    ```python
    from juris_agent import ask

    a = ask("被辞退有补偿吗？")
    ```

    参数原样转给 `JurisAgent.ask`（`session_id` / `history` / `on_delta` …）。
    """
    return get_agent().ask(question, **kwargs)


def reset() -> None:
    """丢掉默认 agent 并关闭它。**测试之间清理、或者改完配置要让新配置生效时用。**"""
    global _default
    with _default_lock:
        agent, _default = _default, None
    if agent is not None:
        agent.close()


__all__ = ["Answer", "Citation", "JurisAgent", "NotReady", "ask", "get_agent", "reset"]
