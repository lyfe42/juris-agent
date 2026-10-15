"""服务的装配：把重量级的、进程级共享的东西造一次，之后所有请求复用（F29）。

## 「进程级」和「请求级」的界线在哪

这条线划错，会出现两类很难查的 bug：

| 东西 | 级别 | 划错的后果 |
|---|---|---|
| 检索器（含 bge 模型，约 1GB 显存） | **进程级** | 每请求加载一次 → 首个 token 前等十几秒，显存抖爆 |
| 工具执行器（含熔断器、幂等缓存） | **进程级** | 每请求重建 → **熔断器永远学不到东西**（每次都是空窗口），幂等缓存命中率永远是 0 |
| LLM 客户端（含连接池） | **进程级** | 每请求新建 → 连接池白建，每次都要重新握手 |
| 记忆管理器 | **请求/会话级** | 共享 → A 用户的月薪出现在 B 用户的答案里 |
| 流式回调 | **请求级** | 共享 → A 的答案流进 B 的页面 |

前四条在 `graph.py` 的 `build_graph` docstring 里说过一遍；
后两条是接 `api/` 时才出现的，所以在这里再说一次——
**这条线不是"配置项"，是这个系统能不能正确工作的前提。**

## 为什么启动时允许失败

`build_runtime()` **不会**因为"没有 API key"或"没有装 GPU 模型"就抛异常；
它把问题记进 `problems`，`ready=False`，然后照样把服务起起来：
`/api/health` 会如实报告缺什么，`/api/chat` 会给出一条明确的错误。

反过来的做法（缺东西就拒绝启动）看着"更严谨"，实际更糟：

- 演示现场想看一下**前端长什么样**、想验证一下**引用能不能点开**，
  都得先把 2.5GB 的模型装完才行；
- 容器编排里"启动失败 → 重启 → 又失败"变成一个转圈的崩溃循环，
  而真正的原因（少一个环境变量）淹没在重启日志里。

**降级要"看得见"，不要"藏起来"。** 所以 `problems` 是一等公民：
它在 `/api/health` 里出现，也在每个聊天的响应里出现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from juris_agent.api.sessions import SessionStore
from juris_agent.config import Settings, get_settings
from juris_agent.observability import get_logger

log = get_logger(__name__)


@dataclass
class Runtime:
    """一颗进程级的"依赖树"，以及它的健康状况。"""

    settings: Settings
    sessions: SessionStore
    runner: Any = None
    """`AgentRunner`。为 `None` 表示装配失败了——见 `problems`。"""

    retriever: Any = None
    llm: Any = None
    started_at: float = field(default_factory=time.time)
    problems: list[str] = field(default_factory=list)
    """**装配过程中真实遇到的每一个问题。**空列表才代表一切正常。"""

    @property
    def ready(self) -> bool:
        """能不能真的回答一个劳动法问题。

        ⚠️ **这和"服务起没起来"是两件事。** 服务总是能起来（见模块 docstring），
        但只有 runner 存在时才答得出东西。把这两个状态合成一个布尔值的话，
        前端就分不清"后端挂了"和"后端在跑但缺配置"——
        而这两种情况下用户该采取的行动完全不同。
        """
        return self.runner is not None

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ready": self.ready,
            "problems": list(self.problems),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "llm_mode": self.settings.llm_mode,
            "model_gen": self.settings.model_gen,
            "retrieval": {
                "chunker": self.settings.retrieval.chunker,
                "rewrite": self.settings.retrieval.rewrite,
                "use_rerank": self.settings.retrieval.use_rerank,
                "inject_k": self.settings.retrieval.inject_k,
            },
            "memory": {
                "enabled": self.settings.memory.enabled,
                "policy": self.settings.memory.policy,
                "total_tokens": self.settings.memory.total_tokens,
            },
            "sessions": self.sessions.stats(),
        }

    def close(self) -> None:
        """关掉持有资源的东西。

        **`LLMClient` 是唯一真的需要关的**（它持有一个 httpx 连接池）。
        检索器不需要：它的资源就是几个 numpy 数组和一份模型权重，
        进程退出时由解释器回收——写一个"关闭检索器"的方法只会让人以为
        那里有什么必须显式释放的东西。
        """
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()


def build_runtime(
    settings: Settings | None = None,
    *,
    runner: Any = None,
    retriever: Any = None,
    llm: Any = None,
    new_memory: Any = None,
) -> Runtime:
    """装配。**所有参数都是可注入的，这不是为了灵活，是为了能测。**

    一个只有"从零装配真实依赖"这一条路的服务，测试里就只剩两个选择：
    ① 真的加载模型 + 真的调 API（慢、要钱、不可复现）；
    ② 不测。

    注入点让测试能塞一个假检索器和假模型进去，把**接口契约**
    （事件流对不对、会话有没有串、CAS 拦不拦得住）测干净。
    这和外层 Agent 测试用 `ScriptedLLM` 是同一个理由。
    """
    s = settings or get_settings()
    sessions = SessionStore(new_memory=new_memory)
    problems: list[str] = []

    # --- 检索器：按需装配 ------------------------------------------------- #
    if retriever is None:
        try:
            from juris_agent.retrieval.build import build_default_retriever

            retriever = build_default_retriever(settings=s)
        except Exception as e:  # noqa: BLE001
            # 捕获所有异常是**刻意的**：这一段的失败原因五花八门
            # （没装 `ml` extra、没下载模型、语料/索引不存在、显存不够），
            # 而每一种的处理方式完全一样——记下来、降级、继续起来。
            # 让它们各自抛出去，只会让服务在这四种情况下的行为不一致。
            problems.append(f"检索器未装配：{type(e).__name__}: {e}")

    # --- LLM 客户端 ------------------------------------------------------ #
    if llm is None:
        try:
            from juris_agent.llm import LLMClient

            llm = LLMClient()
        except Exception as e:  # noqa: BLE001
            problems.append(f"LLM 未装配：{type(e).__name__}: {e}")

    # --- 组装 runner ------------------------------------------------------ #
    if runner is None and (retriever is not None or llm is not None):
        try:
            from juris_agent.agent.graph import make_runner

            runner = make_runner(retriever=retriever, llm=llm, settings=s)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Agent 未装配：{type(e).__name__}: {e}")

    if runner is None and not problems:
        problems.append("未提供任何依赖，服务只能返回健康信息")

    rt = Runtime(settings=s, sessions=sessions, runner=runner, retriever=retriever, llm=llm,
                 problems=problems)
    if problems:
        # **降级要吼一声。** 不吼的话，"服务起来了但答不了问题"
        # 会在演示现场才被发现。
        log.warning("服务降级启动", problems=problems, ready=rt.ready)
    else:
        log.info("服务就绪", model=s.model_gen, chunker=s.retrieval.chunker)
    return rt


__all__ = ["Runtime", "build_runtime"]
