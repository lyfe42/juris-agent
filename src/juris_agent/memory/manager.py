"""记忆管理器：四层抢同一份 token 预算，谁先被砍（F21）。★

## 这一段在回答什么问题

前三层各自都"想多要一点"：

- 槽位想说"我十几个字段全给，反正才两百 token"
- 窗口想说"保留 20 轮原文最完整"
- 摘要想说"每 3 轮压一段，信息损失最小"
- 长期想说"每次捞 10 条，总有一条用得上"

四层全满足 = 回到起点（上下文又被塞满了）。所以必须有人做**分配**，
而且要**说清楚砍谁、为什么**。

## 分配顺序，以及为什么是这个顺序

**判据不是"哪层更常见"，而是"哪层错了后果最严重"。**

| 顺序 | 层 | 砍掉它的后果 | 能不能省 |
|---|---|---|---|
| 1 | 槽位 | **算错钱**。用户拿到一个数字，看不出来是错的 | ❌ 绝不省 |
| 2 | 近期原文 | 答非所问。用户能立刻发现"它没听懂我在问什么" | 省到只剩最后 1 轮为止 |
| 3 | 摘要 | 丢掉背景。答案变浅，但仍然正确 | ✅ 优先砍 |
| 4 | 长期召回 | 没有影响。它本来就是"锦上添花" | ✅ 最先砍 |

这张表里最该记住的是第 1 行和第 3 行的对比：**摘要和槽位看起来都是
"压缩过的历史"，但一个能砍一个不能砍**——因为它们的失真后果不一样。
摘要失真 = 少知道一件事；槽位失真 = **答错**。

## 一个必须守住的下限

**近期窗口至少保留最后 1 轮，哪怕它本身就超了预算。**

理由：没有"这一轮的问题"，就没有对话。宁可让 prompt 整体超一点
（超了由上层截断法条上下文，那是另一层预算），也不能把问题本身丢掉。
这条写在 `_allocate()` 里，有单测钉着。

## 四策略是怎么落在这份代码上的

F22 要对比四种记忆策略，它们**不是四份实现，而是同一个管理器换参数**：

| 策略 | 槽位 | 窗口 | 摘要 | 长期 |
|---|---|---|---|---|
| `no_memory` | ✗ | 只留最后 1 轮 | ✗ | ✗ |
| `sliding_window` | ✗ | 全部预算给窗口 | ✗ | ✗ |
| `sliding_window_summary` | ✗ | 窗口 | ✓ | ✗ |
| `layered`（本项目的方案） | ✓ | 窗口 | ✓ | ✓ |

**这是刻意的**：如果四种策略是四份代码，那对比出来的差异里就混进了
"实现水平的差异"，而不是"策略的差异"。同一份代码换开关，才是干净的对照实验。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from juris_agent.memory.longterm import LongTermStore, MemoryItem
from juris_agent.memory.slots import Slots
from juris_agent.memory.tokens import count_tokens
from juris_agent.memory.working import Message, Summary, WorkingMemory, window_tokens


class MemoryPolicy(str, Enum):
    """四种记忆策略。**顺序不是随意的**：从上到下是"记忆能力递增"，
    F22 的保持率曲线就是按这个顺序画的，曲线应该单调上升——
    **如果不单调，说明实现里有 bug，而不是"策略之间没有好坏"**。"""

    NO_MEMORY = "no_memory"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_SUMMARY = "sliding_window_summary"
    LAYERED = "layered"


class MemoryBudget(BaseModel):
    """一份预算怎么分。

    **全部给绝对值，不给百分比。**百分比看着优雅，但排查问题时
    "摘要层拿了 34%（712 token）"这种信息没法用——
    你要的是"摘要层只拿到 1 段，因为剩下的预算只够 1 段"。
    """

    total: int = 2048
    """记忆层的总预算。选 2048 的理由：答案生成的 `max_tokens` 是 3072，
    prompt 侧留两倍于输出的空间，在 1M 上下文的模型上非常宽裕——
    **所以这个数不是为了"塞得下"，是为了"够便宜、够聚焦"。**
    调大它不会让答案更好（这是要在报告里用实验说明的一件事）。"""
    slots_max: int = 400
    """槽位软上限。十几个字段大约 200 token，400 是两倍余量。
    **超了不是直接砍，而是先砍"当前用不到的槽位"**（见 `_allocate_slots`）。"""
    recent_min_turns: int = 1
    """近期窗口的硬下限。见模块 docstring「一个必须守住的下限」。"""


class LayerUse(BaseModel):
    """某一层实际用了多少、放了几条。**报表里的每一行就是它。**"""

    name: str
    tokens: int = 0
    items: int = 0
    detail: str = ""


class MemoryBundle(BaseModel):
    """交给上层（`agent`）的成品：一段可以直接塞进 prompt 的文本 + 一份账。"""

    prompt_block: str = ""
    layers: list[LayerUse] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    """被砍掉的东西 + 为什么。**这份清单是 F22 报告的核心素材**——
    "第 20 轮丢了信息"能一眼看出"是因为摘要层被砍了"。"""
    policy: str = MemoryPolicy.LAYERED.value
    total_tokens: int = 0

    def layer_map(self) -> dict[str, int]:
        return {lay.name: lay.tokens for lay in self.layers}


class MemoryManager(BaseModel):
    """四层记忆的装配与分配。

    它是**有状态的**（`working` 和 `long_term` 跨轮累积），但每轮的
    `build()` 是纯函数式的：给同样的输入，输出同样的 bundle。
    """

    policy: MemoryPolicy = MemoryPolicy.LAYERED
    budget: MemoryBudget = Field(default_factory=MemoryBudget)
    working: WorkingMemory = Field(default_factory=WorkingMemory)
    long_term: LongTermStore = Field(default_factory=LongTermStore)
    slots: Slots = Field(default_factory=Slots)

    # ------------------------------------------------------------------ #
    # 喂数据
    # ------------------------------------------------------------------ #

    def observe(self, text: str, *, turn: int) -> list[str]:
        """把一句用户话喂进槽位层。**返回这次真正写进去的槽位名。**

        ⚠️ **这是记忆层和对话循环之间唯一需要的那根线**，
        而它一开始是缺的：`build()` 只**读** `self.slots`，没有任何地方
        **写**它。结果就是"管理器造出来了、策略也选好了、槽位永远是空的"——
        而且不报错，只是保持率特别低。

        和 `SlotStore.observe` 转发到同一份逻辑（`Slots.observe`），
        **不另写一遍**：两份必然漂移，漂移的那一份一定是你最少调用的那份。
        """
        return self.slots.observe(text, turn=turn)

    def derive(self, *, turn: int) -> list[str]:
        """跑派生（入职日期 → 工作月数、城市 → 社平工资）。

        **每轮都要调**，而且要在 `observe()` 之后、`build()` 之前——
        派生出来的值这一轮就该能被用上。
        """
        return self.slots.derive(turn=turn)

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def build(
        self,
        question: str,
        history: list[dict[str, str]],
        *,
        current_turn: int,
    ) -> MemoryBundle:
        """把一段历史压成"能给模型看的那一坨"。

        `history` 是**完整**历史（形如 `[{"role": "user", "content": "..."}]`）。
        **刻意做成"每次喂全量"而不是"每轮喂增量"**：

        - 本模块因此变成纯函数，可重放、可单测（不用维护"上一轮我看到了什么"的
          隐藏状态）
        - 代价是每轮要重新切一次词。20 轮 × 几十条记录，代价可以忽略
        - 更重要的是：**增量式的记忆最容易出"漏喂一轮"的 bug，而且不报错**
        """
        msgs = _to_messages(history)

        layers: list[LayerUse] = []
        dropped: list[str] = []

        # ---- ① 槽位：预算最高优先级，**先切走它那一份** ----
        slots_text = self._allocate_slots(layers, dropped)

        # ---- ② 近期原文 ----
        budget_left = self.budget.total - count_tokens(slots_text)
        recent, recent_text = self._allocate_recent(msgs, budget_left, layers, dropped)
        budget_left -= count_tokens(recent_text)

        # ---- ③ 摘要 ----
        summaries_text = ""
        if self.policy in (MemoryPolicy.SLIDING_WINDOW_SUMMARY, MemoryPolicy.LAYERED):
            summaries_text = self._allocate_summaries(msgs, budget_left, layers, dropped)
            budget_left -= count_tokens(summaries_text)

        # ---- ④ 长期召回 ----
        recalled_text = ""
        if self.policy is MemoryPolicy.LAYERED:
            recalled_text, new_items = self._allocate_recall(
                question, msgs, current_turn, budget_left, layers, dropped
            )
            budget_left -= count_tokens(recalled_text)
        else:
            new_items = []
        self._archive(recent, msgs, new_items)

        block = "\n\n".join(t for t in (slots_text, recent_text, summaries_text, recalled_text) if t)
        return MemoryBundle(
            prompt_block=block,
            layers=layers,
            dropped=dropped,
            policy=self.policy.value,
            total_tokens=count_tokens(block),
        )

    # ------------------------------------------------------------------ #
    # 逐层分配
    # ------------------------------------------------------------------ #

    def _allocate_slots(self, layers: list[LayerUse], dropped: list[str]) -> str:
        """① 槽位。**最后一个被砍的层。**

        ⚠️ 超预算时**不是按顺序砍**，而是按"这一轮用不用得到"砍：

        用户问"加班费怎么算"，那么 `know_date`（仲裁时效）、`contract_term_months`
        （试用期）这些槽位这一轮完全用不上——砍掉它们，**用户一个数字都不会少**。
        反过来，如果按"后来加的槽位先砍"这种实现细节顺序砍，
        被砍掉的那个可能恰好是这次要用的。

        判据用 `question` 里出现的关键词反查最合适，但那需要每个槽位配关键词。
        这里用一个更省事、也更稳的近似：**按"上一次被更新/引用的轮次"由远及近砍**，
        因为最近用到的槽位就是这一轮最可能要用的。
        """
        if self.policy not in (MemoryPolicy.LAYERED,):
            return ""

        text = self.slots.prompt_block()
        if not text:
            layers.append(LayerUse(name="slots", detail="空"))
            return ""
        n = count_tokens(text)
        if n <= self.budget.slots_max:
            layers.append(
                LayerUse(name="slots", tokens=n, items=len(self.slots.values), detail="全部")
            )
            return text

        # 超预算：按 turn 由新到旧保留，直到装得下
        ordered = sorted(self.slots.values.items(), key=lambda kv: -kv[1].turn)
        kept: dict[str, Any] = {}
        used = 0
        for name, v in ordered:
            line_cost = count_tokens(f"- {name}：{v.value}") + 2
            if used + line_cost > self.budget.slots_max:
                dropped.append(f"槽位 {name}（预算不足，且是较早轮次给的）")
                continue
            kept[name] = v
            used += line_cost

        trimmed = Slots(values=kept, history=self.slots.history)
        text = trimmed.prompt_block()
        n2 = count_tokens(text)
        layers.append(
            LayerUse(
                name="slots",
                tokens=n2,
                items=len(kept),
                detail=f"超软上限，保留最近的 {len(kept)}/{len(self.slots.values)} 个槽位",
            )
        )
        return text

    def _allocate_recent(
        self,
        msgs: list[Message],
        budget_left: int,
        layers: list[LayerUse],
        dropped: list[str],
    ) -> tuple[list[Message], str]:
        """② 近期原文。**从最近往回填**，直到预算用完或窗口用完。

        ⚠️ 总预算为负（槽位就吃超了）时仍然保留最后 1 轮——
        见模块 docstring「一个必须守住的下限」。
        """
        if self.policy is MemoryPolicy.NO_MEMORY:
            # 无记忆策略：只留最后一轮。**这不是"没记忆"的字面实现**——
            # 一个连问题都不带的 prompt 连跑都跑不起来，那就不是策略对比，
            # 是拿一个跑不通的基线去比。真正的"无记忆"是"只有这一轮"。
            r = self.working.ingest(msgs)
            recent = r["recent"][-2:] if r["recent"] else []
            text = _render_recent(recent)
            layers.append(
                LayerUse(name="recent", tokens=count_tokens(text), items=len(recent), detail="仅最后一轮")
            )
            return recent, text

        if self.policy is MemoryPolicy.SLIDING_WINDOW:
            # 窗口策略：**不压缩，把记忆预算全给原文**。
            # 于是"window"这个参数在策略对比里被拉大——这正是它和 layered 的区别。
            keep = max(self.working.window, self.budget.total // 200)
            self.working.window = keep
        r = self.working.ingest(msgs)
        recent = list(r["recent"])

        # 从最近往回填
        kept: list[Message] = []
        used = 0
        for m in reversed(recent):
            cost = window_tokens([m])
            if kept and used + cost > budget_left:
                dropped.append(
                    f"近期原文：第 {m.turn} 轮（预算只剩 {max(0, budget_left - used)} token）"
                )
                break
            kept.append(m)
            used += cost
        kept.reverse()

        # 硬下限：至少一轮。**超过预算也保**，理由见模块 docstring。
        if not kept and recent:
            kept = [recent[-1]]
            dropped.append("近期原文：只保住了最后 1 轮（预算被高优先级层吃满）")

        text = _render_recent(kept)
        layers.append(
            LayerUse(
                name="recent",
                tokens=count_tokens(text),
                items=len(kept),
                detail=f"窗口 {self.working.window} 轮，实际保留 {len(kept)} 条消息",
            )
        )
        return kept, text

    def _allocate_summaries(
        self,
        msgs: list[Message],
        budget_left: int,
        layers: list[LayerUse],
        dropped: list[str],
    ) -> str:
        """③ 摘要。**从最新的一段往回放**——

        越近的摘要越可能和这一轮相关。放不下的直接丢，不截断（截断一段摘要
        会得到一个半截句子，模型会去猜后面是什么）。
        """
        self.working.ingest(msgs)  # ingest 会把该压的压掉
        if not self.working.summaries:
            layers.append(LayerUse(name="summaries", items=0, detail="还没有可压的历史"))
            return ""

        kept: list[Summary] = []
        used = 0
        for s in reversed(self.working.summaries):
            cost = count_tokens(s.render()) + 2
            if used + cost > budget_left:
                dropped.append(
                    f"摘要 第{s.from_turn}-{s.to_turn}轮（{cost} token 装不下）"
                )
                continue
            kept.append(s)
            used += cost
        kept.reverse()
        text = "\n".join(s.render() for s in kept)
        layers.append(
            LayerUse(
                name="summaries",
                tokens=count_tokens(text),
                items=len(kept),
                detail=f"共 {len(self.working.summaries)} 段，放进 {len(kept)} 段",
            )
        )
        return text

    def _allocate_recall(
        self,
        question: str,
        msgs: list[Message],
        current_turn: int,
        budget_left: int,
        layers: list[LayerUse],
        dropped: list[str],
    ) -> tuple[str, list[MemoryItem]]:
        """④ 长期召回。**第一个被砍的层**——它本来就是可选的。"""
        if budget_left <= 0:
            layers.append(LayerUse(name="recall", detail="预算已用尽，跳过"))
            dropped.append("长期召回：预算用尽（这一层优先级最低，先砍它是对的）")
            return "", []

        hits = self.long_term.search(question, current_turn=current_turn, k=3)
        if not hits:
            layers.append(
                LayerUse(name="recall", items=0, detail="没有词重叠的历史，空手而归")
            )
            return "", []

        kept: list[MemoryItem] = []
        used = 0
        for item in hits:
            cost = count_tokens(item.render()) + 2
            if used + cost > budget_left:
                dropped.append(f"长期召回 第{item.turn}轮（{cost} token 装不下）")
                continue
            kept.append(item)
            used += cost
        if not kept:
            return "", []
        text = "【可能相关的历史（按需召回，不一定都对本题有用）】\n" + "\n".join(
            i.render() for i in kept
        )
        layers.append(
            LayerUse(name="recall", tokens=count_tokens(text), items=len(kept), detail="按需召回")
        )
        return text, kept

    def _archive(
        self, recent: list[Message], msgs: list[Message], new_items: list[MemoryItem]
    ) -> None:
        """把**离开窗口的轮次**存进长期层。

        为什么把这件事放在 `build()` 的末尾而不是 `add()` 里：
        长期层的内容取决于"这一轮窗口保留到哪里"，而那是每轮变化的。
        由 `build()` 统一决定，长期层就永远不需要知道窗口的参数——
        **责任边界按"谁掌握判断所需的信息"来划。**
        """
        if self.policy is not MemoryPolicy.LAYERED:
            return
        in_window = {m.turn for m in recent}
        already = {i.turn for i in self.long_term.items}
        for m in msgs:
            if m.role != "user" or m.turn in in_window or m.turn in already:
                continue
            answer = next(
                (x.content for x in msgs if x.turn == m.turn and x.role == "assistant"), ""
            )
            self.long_term.add(MemoryItem(turn=m.turn, question=m.content, answer=answer))


def _to_messages(history: list[dict[str, str]]) -> list[Message]:
    """把 `[{"role": ..., "content": ...}]` 转成带**轮次号**的消息列表。

    ## ⚠️ 轮次是"一轮"，不是"一条消息"

    踩过的坑：一开始用的是 `enumerate(history, start=1)`，也就是
    **一条消息一个轮次号**。后果是"窗口 6 轮"实际上只保住了**3 轮对话**
    （一轮里用户和助手各算一条）——而所有文档、所有报表说的都是"6 轮"。
    这种错**不会报错**，只会让"第 20 轮丢了信息"的排查全对着错的地方找。

    正确的一轮 = 一条用户消息 + 它后面的助手回复。所以轮次号**只在遇到
    用户消息时加一**，助手消息沿用当前轮次。

    历史以助手消息开头（少见但可能出现：历史被裁剪过）时，按第 1 轮算，
    不抛异常——**记忆层不该因为输入形状少见就整条链路挂掉**。
    """
    msgs: list[Message] = []
    turn = 0
    for m in history:
        role = str(m.get("role", "user"))
        if role not in ("user", "assistant"):
            continue
        if role == "user":
            turn += 1
        elif turn == 0:
            turn = 1
        msgs.append(Message(role=role, content=str(m.get("content", "")), turn=turn))
    return msgs


def _render_recent(msgs: list[Message]) -> str:
    if not msgs:
        return ""
    lines = ["【最近的对话（原文，未压缩）】"]
    for m in msgs:
        who = "用户" if m.role == "user" else "助手"
        lines.append(f"{who}：{m.content}")
    return "\n".join(lines)


__all__ = ["LayerUse", "MemoryBudget", "MemoryBundle", "MemoryManager", "MemoryPolicy"]
