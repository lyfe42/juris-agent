"""会话（F29 + F32 里那一半并发问题）。

## 会话里存什么，为什么是这三样

| 存的东西 | 干什么用 | 不存会怎样 |
|---|---|---|
| `history` | 喂给记忆层和历史窗口的原文 | 第 2 轮就变成"全新对话"，用户得重复一遍 |
| `memory` | 槽位表 + 摘要 + 长期召回（见 `memory/manager.py`） | 每轮重抽一遍历史（几十毫秒），且**段摘要是空重算的** |
| `version` | 防止并发写覆盖（下面详述）| 见「为什么要版本号」 |

## 为什么要锁，**而且**为什么要版本号

这两个东西防的是**不同的**问题，缺一个都会留下真实的洞：

**锁（`asyncio.Lock`）防的是"同时"。** 同一个会话的两个请求同时进来
（用户连点两次发送、或者前端重试了），如果不串行化，两轮会同时读到
「历史只有 3 条」，各自算完写回，**后写的把先写的整个盖掉**——
用户看到的是"我说的话消失了"。表现是随机的，日志里也看不出来。

**版本号（CAS，读的时候记下版本、写之前确认没变）防的是"先后但没人知道"。**
锁只能防住**同一个进程里**的并发。这三种情况它一点用都没有：

1. 多进程部署（`uvicorn --workers 4`）——锁是进程内的，四个进程各锁各的；
2. 会话状态以后挪到 Redis / 数据库——锁在本地内存，跨机器无效；
3. **代码里某条新路径忘了加锁**——这是最常见的一种，而且**不会报错**。

版本号的写法是：读的时候拿到 `version=7`，写完之前调
`commit(expected_version=7)`；如果这时会话已经是 8 了，
说明中间有人改过，**抛 `SessionConflict`**，由调用方决定重试还是报错。
**它把"静默覆盖"变成了一条明确的错误**——这是它唯一的、也是全部的价值。

> 面试被问「并发怎么写」时，能答出"锁和 CAS 防的不是同一件事，
> 而且我用的是**进程内锁 + 版本号**、所以多进程下也不会静默覆盖"，
> 和答"我加了个锁"不是一回事。

## 内存会涨吗

会。所以有 `max_sessions` 和 TTL，超了就按"最久没用"淘汰（LRU）。
**淘汰掉的是"记忆"，不是"错误"**：用户隔了很久再说话，
就是一段新会话，服务端不需要为此报错。
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from juris_agent.observability import get_logger

log = get_logger(__name__)


class SessionConflict(RuntimeError):
    """会话在"读完"和"写回"之间被别人改过。

    **不要吞掉它。** 吞掉等于回到"后写的盖掉先写的"那个 bug，
    而且是静默的。要么让调用方重试一次，要么把这轮判为失败让用户重发——
    两个选择都比悄悄丢一轮对话好。
    """


@dataclass
class Session:
    """一段对话。**一个会话一个实例，绝不在会话之间共享**（见 `memory/manager.py`）。"""

    session_id: str
    memory: Any = None
    history: list[dict[str, str]] = field(default_factory=list)
    version: int = 0
    """每次成功写回 +1。**CAS 的凭据。**"""
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """**每个会话一把锁**，不是全服务一把。

    全服务一把锁的话，一个用户在算经济补偿，另一个用户的简单问答也要排队——
    并发度直接退化成 1。锁的粒度必须和"被保护的东西"一样大。"""

    @property
    def turn(self) -> int:
        """这段会话已经聊到第几轮（用户发过几句话）。"""
        return sum(1 for m in self.history if m.get("role") == "user")

    def messages(self) -> list[dict[str, str]]:
        """历史的一份**拷贝**。

        ⚠️ 必须拷贝。直接把手上的 list 交出去，图里的代码一旦
        往里 append（哪怕只是想加一条当前问题），就会**改到会话状态本身**——
        然后下一轮的历史里会多出一条重复的用户消息。这种 bug 表现为
        "模型好像看到了两遍同一个问题"，从日志上完全看不出原因。
        """
        return [dict(m) for m in self.history]


class SessionStore:
    """会话表的实现：**进程内、LRU 淘汰、每会话一把锁 + 版本号**。

    ⚠️ 它**不是**分布式会话存储，也刻意没做成那样：
    多实例部署时正确做法是换成 Redis（版本号那套逻辑原样能用，
    因为 CAS 本来就是为"共享存储 + 无共享锁"设计的）。
    现在做成内存版，是在**如实反映当前部署形态**——
    写一套没人用的分布式会话代码，比重写它更糟。
    """

    def __init__(
        self,
        *,
        max_sessions: int = 500,
        ttl_seconds: float = 3600.0,
        new_memory: Callable[[], Any] | None = None,
    ) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._new_memory = new_memory
        # `OrderedDict` + `move_to_end` = LRU。用它而不是 `dict`：
        # 淘汰要按"最久没用"来，而普通 dict 的插入顺序和"最近使用"没关系。
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    # -- 取 ------------------------------------------------------------------ #

    def _make_memory(self) -> Any:
        if self._new_memory is not None:
            return self._new_memory()
        from juris_agent.memory.manager import manager_from_settings

        return manager_from_settings(None)

    def new_id(self) -> str:
        """新会话 id。**前缀 `s-` 是为了在日志里一眼认出来**——
        `s-3f9a2b1c` 和 `req` 里的请求 id 混在一起时，
        没有前缀你要盯着看半天才知道哪个是哪个。"""
        return f"s-{secrets.token_hex(8)}"

    def get_or_create(self, session_id: str | None) -> Session:
        """拿到会话，没有就建一个。**同步方法，故意不加锁。**

        为什么不加：它只做"读写一个 dict"，在 asyncio 里没有 `await` 就不会被切走，
        **本身就是原子的**。为它包一层锁只会让人以为"这里真的会有并发问题"，
        从而在别处放松警惕。
        """
        self.sweep()
        if session_id and session_id in self._sessions:
            s = self._sessions[session_id]
            s.last_used = time.monotonic()
            self._sessions.move_to_end(session_id)
            return s

        sid = session_id or self.new_id()
        s = Session(session_id=sid, memory=self._make_memory())
        self._sessions[sid] = s
        self._sessions.move_to_end(sid)
        log.info("新建会话", session_id=sid)
        # ⚠️ **建完再清一次**，不能只靠上面开头那次。
        # 开头那次清完、这里又塞进来一个，于是"调用方拿到东西之后
        # 表里有几个"可能会**超过 max_sessions**（刚好超一个）。
        # 只多一个看着无所谓，但 `max_sessions` 是个承诺：
        # 它写着 500，实际能到 501，那压测和容量估算就是错的。
        # **对"上限"这种数，差一个和差一千个是同一类问题。**
        self.sweep()
        return s

    # -- 写 ------------------------------------------------------------------ #

    def commit(self, session: Session, *, expected_version: int, question: str, answer: str) -> None:
        """把这一轮写回会话。版本号不对就抛 `SessionConflict`。

        ## 为什么 `expected_version` 是必传参数（而不是从 `session` 上现读）

        因为那样就**等于没有校验**：从 `session` 上读出来的版本号永远是"当前的"，
        永远对得上。版本号的全部意义在于"**我读的时候是几**"，
        而那个数只有调用方手里有。所以这里强制它显式传进来——
        **签名本身就把这个约束钉死了，比写在注释里可靠。**
        """
        if session.version != expected_version:
            raise SessionConflict(
                f"会话 {session.session_id} 在读取后被改动过"
                f"（读到时 version={expected_version}，现在是 {session.version}）。"
                "这一轮没有写回，以免覆盖掉别人的改动。"
            )
        session.history.append({"role": "user", "content": question})
        session.history.append({"role": "assistant", "content": answer})
        session.version += 1
        session.last_used = time.monotonic()
        self._sessions.move_to_end(session.session_id)

    # -- 清理 ---------------------------------------------------------------- #

    def sweep(self) -> int:
        """淘汰过期和超量的会话，返回淘汰了几个。

        **在 `get_or_create` 里顺手调用**，不另起后台任务：
        一个只为了清内存的定时任务，在有流量的服务里是多余的
        （有流量就一定会走到这里）；而在没流量的服务里它又白跑。
        顺手清是这两者都成立的唯一做法。
        """
        now = time.monotonic()
        dead = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_used > self.ttl_seconds
        ]
        for sid in dead:
            del self._sessions[sid]

        # LRU 超量淘汰：`OrderedDict` 的头部是最久没用的。
        while len(self._sessions) > self.max_sessions:
            sid, _ = self._sessions.popitem(last=False)
            dead.append(sid)

        if dead:
            log.info("淘汰会话", count=len(dead), remaining=len(self._sessions))
        return len(dead)

    def stats(self) -> dict[str, Any]:
        """给 `/api/health` 和压测用。"""
        return {
            "sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> None:
        """清空。**测试用**，也是"我不信任当前状态、重来一遍"的入口。"""
        self._sessions.clear()


__all__ = ["Session", "SessionConflict", "SessionStore"]
